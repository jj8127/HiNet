import re
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logging')

def parse_log(path: str) -> dict:
    base = os.path.basename(path)
    # 간단한 라벨: epXX_cYY (XX=epoch, YY=calib step)
    epoch = re.search(r'ep(\d+)', base)
    calib = re.search(r'calib(\d+)', base)
    label = f"ep{epoch.group(1)}_c{calib.group(1)}" if epoch and calib else base.replace('.log','')
    bits = 4 if '4bit' in base else 8
    last_psnr_s, last_psnr_r = None, None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re.search(r'TEST:\s+PSNR_S:\s+([-\d\.]+)\s+\|\s+PSNR_C:\s+([-\d\.]+)', line)
            if m:
                last_psnr_s = float(m.group(1))
                last_psnr_r = float(m.group(2))
    return {'label': label, 'bits': bits, 'PSNR_S': last_psnr_s, 'PSNR_C': last_psnr_r}

def main():
    log_dir = LOG_DIR
    logs = []
    for log_path in glob.glob(os.path.join(log_dir, '*.log')):
        result = parse_log(log_path)
        if result['PSNR_S'] is not None:
            logs.append(result)
    df = pd.DataFrame(logs)
    for bits in sorted(df['bits'].unique()):
        sub = df[df['bits'] == bits]
        fig, ax = plt.subplots(figsize=(8,5))
        indices = range(len(sub))
        width = 0.35
        ax.bar([i-width/2 for i in indices], sub['PSNR_S'], width, label='PSNR_S')
        ax.bar([i+width/2 for i in indices], sub['PSNR_C'], width, label='PSNR_C')
        ax.set_xticks(indices)
        ax.set_xticklabels(sub['label'], rotation=30, ha='right', fontsize=11)  # 더 짧고, 기울기 작게, 폰트 약간 키움
        ax.set_title(f"{bits}-bit QAT Experiments")
        ax.set_ylabel("PSNR (dB)")
        # legend를 바깥 우상단으로 이동
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
        # 더 부드러운 격자선
        ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7, color='gray', alpha=0.4)
        plt.tight_layout(rect=[0,0,0.88,1])  # 그래프+범례 공간 확보
        savepath = os.path.join(log_dir, f'qat_psnr_{bits}bit_by_model.png')
        plt.savefig(savepath)
        print(f"[그래프 저장] {savepath}")
        plt.show()

if __name__ == '__main__':
    main()
