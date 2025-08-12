# 경로를 '여기서' 설정합니다. (명령줄 인자 없음)
# [모드 A] 특정 세션/로그를 직접 지정
SESSION_DIR = '/root/Desktop/HiNet/logging/qat_qbackend_safe_20250812_161125_ep50_calib128'  # 예: "/root/Desktop/HiNet/logging/qat_qbackend_safe_20250812_142025_ep5_calib128"
LOG_PATH    = None  # 예: "/root/Desktop/HiNet/logging/.../train.log"  (SESSION_DIR보다 우선)
OUT_PATH    = None  # 예: "/root/Desktop/HiNet/metrics_from_log.png" (미지정 시 <세션>/metrics_from_log.png로 저장)

# [모드 B] 최신 세션 자동 탐색 (A를 쓰면 무시)
AUTO_FIND_LAST = True
ROOT_LOG_DIR   = "logging"   # 최신 세션을 찾을 상위 폴더

# ------------------------------------------------------------
# 아래부터는 수정 불필요
import os, re, json, glob, math

def is_empty_history(d):
    if not isinstance(d, dict):
        return True
    keys = ("epoch","loss","psnr_c","psnr_r")
    has_any = False
    for k in keys:
        v = d.get(k, [])
        if isinstance(v, list) and len(v) > 0:
            has_any = True
            break
    return not has_any

def load_history_json(session_dir):
    hist_path = os.path.join(session_dir, "history.json")
    if not os.path.isfile(hist_path):
        return None
    try:
        with open(hist_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        # 값이 비어있으면 사용하지 않음 (로그 파싱으로 폴백)
        if is_empty_history(j):
            return None
        # 키 가드 & 정렬
        out = {
            "epoch": list(j.get("epoch", [])),
            "loss":  list(j.get("loss", [])),
            "psnr_c": list(j.get("psnr_c", [])),
            "psnr_r": list(j.get("psnr_r", [])),
        }
        return out
    except Exception:
        return None

def parse_train_log(log_path):
    # 예시 라인:
    # 25-08-12 14:13:32 - INFO: TRAIN: done | loss 0.0135
    # 25-08-12 14:13:36 - INFO: EVAL | PSNR_C 37.322 dB | PSNR_S 38.831 dB
    # (PSNR_R 라벨로 출력되는 경우도 지원)
    loss_pat = re.compile(r"TRAIN:\s*done\s*\|\s*loss\s*([0-9eE\.\-]+)")
    eval_cs  = re.compile(r"EVAL\s*\|\s*PSNR_C\s*([0-9eE\.\-]+)\s*dB\s*\|\s*PSNR_S\s*([0-9eE\.\-]+)\s*dB")
    eval_cr  = re.compile(r"EVAL\s*\|\s*PSNR_C\s*([0-9eE\.\-]+)\s*dB\s*\|\s*PSNR_R\s*([0-9eE\.\-]+)\s*dB")

    losses, psnr_c, psnr_r = [], [], []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = loss_pat.search(line)
            if m:
                try:
                    losses.append(float(m.group(1)))
                except Exception:
                    pass
            m = eval_cs.search(line)
            if m:
                try:
                    psnr_c.append(float(m.group(1)))
                    psnr_r.append(float(m.group(2)))
                except Exception:
                    pass
            else:
                m = eval_cr.search(line)
                if m:
                    try:
                        psnr_c.append(float(m.group(1)))
                        psnr_r.append(float(m.group(2)))
                    except Exception:
                        pass

    n = max(len(losses), len(psnr_c), len(psnr_r))
    if n == 0:
        raise RuntimeError(f"로그에서 유효한 항목을 찾지 못했습니다: {log_path}")

    return {
        "epoch": list(range(1, n + 1)),
        "loss": losses,
        "psnr_c": psnr_c,
        "psnr_r": psnr_r,
    }

def find_latest_session(root_dir):
    # train.log 수정 시간이 최신인 세션 선택
    pattern = os.path.join(root_dir, "qat_qbackend_safe_*", "train.log")
    cands = glob.glob(pattern)
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return os.path.dirname(cands[0])

def resolve_paths():
    # 우선순위: LOG_PATH 직접 지정 > SESSION_DIR 지정 > 자동탐색
    if LOG_PATH:
        if not os.path.isfile(LOG_PATH):
            raise FileNotFoundError(f"LOG_PATH not found: {LOG_PATH}")
        session_dir = os.path.dirname(os.path.abspath(LOG_PATH))
        out_path = OUT_PATH or os.path.join(session_dir, "metrics_from_log.png")
        return session_dir, LOG_PATH, out_path

    if SESSION_DIR:
        log_path = os.path.join(SESSION_DIR, "train.log")
        if not os.path.isfile(log_path):
            raise FileNotFoundError(f"train.log not found under SESSION_DIR: {log_path}")
        out_path = OUT_PATH or os.path.join(SESSION_DIR, "metrics_from_log.png")
        return SESSION_DIR, log_path, out_path

    if AUTO_FIND_LAST:
        session_dir = find_latest_session(ROOT_LOG_DIR)
        if not session_dir:
            raise FileNotFoundError(f"No train.log found under: {ROOT_LOG_DIR}")
        log_path = os.path.join(session_dir, "train.log")
        out_path = OUT_PATH or os.path.join(session_dir, "metrics_from_log.png")
        return session_dir, log_path, out_path

    raise RuntimeError("경로 설정 필요: SESSION_DIR 또는 LOG_PATH 지정하세요.")

def _pad(arr, n):
    arr = list(arr)
    if len(arr) < n:
        arr += [float("nan")] * (n - len(arr))
    return arr[:n]

def plot_history(hist, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = max(
        len(hist.get("epoch", [])),
        len(hist.get("loss", [])),
        len(hist.get("psnr_c", [])),
        len(hist.get("psnr_r", [])),
        1,
    )
    ep     = _pad(hist.get("epoch", []), n) or list(range(1, n + 1))
    loss   = _pad(hist.get("loss", []), n)
    psnr_c = _pad(hist.get("psnr_c", []), n)
    psnr_r = _pad(hist.get("psnr_r", []), n)

    # 예시 이미지와 유사한 레이아웃/스타일
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=140, sharex=True)
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[0].plot(ep, psnr_c, linewidth=2)
    axes[0].set_title("PSNR_C"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("dB")

    axes[1].plot(ep, psnr_r, linewidth=2, color="red")
    axes[1].set_title("PSNR_S"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("dB")

    axes[2].plot(ep, loss, linewidth=2)
    axes[2].set_title("Train Loss"); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Loss")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] saved: {out_path} (points: loss={sum(math.isfinite(x) for x in loss)}, "
          f"psnr_c={sum(math.isfinite(x) for x in psnr_c)}, psnr_s={sum(math.isfinite(x) for x in psnr_r)})")

def main():
    session_dir, log_path, out_path = resolve_paths()
    # history.json이 있으면 쓰되 비어있으면 로그 파싱으로 폴백
    hist = load_history_json(session_dir)
    if hist is None:
        hist = parse_train_log(log_path)
    plot_history(hist, out_path)

if __name__ == "__main__":
    main()
