# inference_time.py
#!/usr/bin/env python3
import os, time, argparse, collections
import torch
from torch.utils.data import DataLoader

import config as c
import modules.Unet_common as common
import datasets
from model import Model, init_model

def load_model_any(model_path: str):
    # 라즈파이 최적화: CPU + qnnpack + 스레드 수
    torch.backends.quantized.engine = "qnnpack"
    torch.set_num_threads(4)
    device = torch.device("cpu")

    model, kind = None, None

    # 1) TorchScript 시도
    try:
        model = torch.jit.load(model_path, map_location=device).eval()
        kind = "ts"
    except Exception:
        model = None

    # 2) nn.Module 전체 저장본 시도
    if model is None:
        try:
            obj = torch.load(model_path, map_location=device)
        except Exception as e:
            raise RuntimeError(f"체크포인트 로드 실패: {e}")

        if isinstance(obj, torch.nn.Module):
            model = obj.eval()
            kind = "full"
        elif isinstance(obj, collections.abc.Mapping):
            # 3) state_dict 계열 처리
            state = None
            if "net" in obj and isinstance(obj["net"], dict):
                state = obj["net"]
            elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
                state = obj["state_dict"]
            elif "model" in obj and isinstance(obj["model"], dict):
                state = obj["model"]
            else:
                # plain state_dict로 가정
                state = obj

            # prefix 정리
            new_state = {}
            for k, v in state.items():
                name = k
                if name.startswith("module.model."):
                    name = name[len("module.model."):]
                elif name.startswith("module."):
                    name = name[len("module."):]
                if name.startswith("model."):
                    name = name[len("model."):]
                new_state[name] = v

            m = Model()
            init_model(m)
            m.load_state_dict(new_state, strict=False)
            model = m.eval()
            kind = "state"
        else:
            raise RuntimeError("지원하지 않는 체크포인트 형식입니다. TorchScript(.pt), nn.Module(.pt), state_dict(dict)을 사용하세요.")

    # 양자화 모듈 존재 확인
    qmods = [m for m in model.modules() if 'quantized' in m.__class__.__module__]
    print(f"[info] model_kind={kind}, device={device}, qengine={torch.backends.quantized.engine}")
    print(f"[info] quantized modules found: {len(qmods)}")
    if len(qmods) == 0:
        print("[warn] 양자화 모듈이 없습니다. 이 실행은 FP32 경로일 가능성이 큽니다.")

    return model, device, kind

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="모델 경로 (TorchScript/nn.Module/state_dict)")
    ap.add_argument("--secret_dir", required=True, help="비밀 이미지 폴더")
    ap.add_argument("--cover_dir",  required=True, help="커버 이미지 폴더")
    ap.add_argument("--max_pairs",  type=int, default=10, help="측정할 샘플 수")
    args = ap.parse_args()

    model, device, kind = load_model_any(args.model)

    # DWT/IWT는 CPU에서 수행
    dwt = common.DWT().to(device)
    iwt = common.IWT().to(device)

    # 비밀/커버 폴더에서 직접 로드 (datasets_pair 아님)
    dataset = datasets.HinetDataset(args.secret_dir, args.cover_dir, datasets.transform_val, c.format_val)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    conceal_times, reveal_times, total_times = [], [], []

    with torch.no_grad():
        for idx, (secret, cover) in enumerate(loader):
            if idx >= args.max_pairs:
                break

            secret = secret.to(device)
            cover  = cover.to(device)

            # ── conceal (forward)
            t0 = time.perf_counter()
            inp_fwd = torch.cat((dwt(cover), dwt(secret)), 1)
            out = model(inp_fwd)                           # TS/FP32 공통
            steg_part = out.narrow(1, 0, 4 * c.channels_in)
            _ = iwt(steg_part)                             # 시간만 측정, 저장 X
            conceal_t = time.perf_counter() - t0

            # ── reveal (backward)
            t1 = time.perf_counter()
            out_z = out.narrow(1, 4 * c.channels_in, out.shape[1] - 4 * c.channels_in)
            back_z = torch.randn_like(out_z)
            inp_rev = torch.cat((steg_part, back_z), 1)

            if kind == "ts":
                # TorchScript가 rev=True를 지원하지 않는 경우가 많아 동일 forward 경로 호출
                rev_feat = model(inp_rev)
            else:
                # 원래 Model()은 rev=True 지원
                try:
                    rev_feat = model(inp_rev, rev=True)
                except TypeError:
                    rev_feat = model(inp_rev)

            secret_rev_part = rev_feat.narrow(1, 4 * c.channels_in, rev_feat.shape[1] - 4 * c.channels_in)
            _ = iwt(secret_rev_part)
            reveal_t = time.perf_counter() - t1

            total_t = conceal_t + reveal_t
            conceal_times.append(conceal_t)
            reveal_times.append(reveal_t)
            total_times.append(total_t)

            print(f"[{idx}] conceal: {conceal_t:.4f}s | reveal: {reveal_t:.4f}s | total: {total_t:.4f}s")

    n = len(total_times)
    if n > 0:
        print("\n--- 평균 ---")
        print(f"conceal_avg: {sum(conceal_times)/n:.4f}s")
        print(f"reveal_avg : {sum(reveal_times)/n:.4f}s")
        print(f"total_avg  : {sum(total_times)/n:.4f}s")
    else:
        print("로드된 샘플이 없습니다. --secret_dir/--cover_dir 및 확장자(c.format_val)를 확인하세요.")

if __name__ == "__main__":
    main()
