import re
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logging')


def parse_log(path: str) -> list[dict]:
    records = []
    bits = 4 if '4bit' in os.path.basename(path) else 8
    calib_match = re.search(r'calib(\d+)', os.path.basename(path))
    calibration = int(calib_match.group(1)) if calib_match else None
    current_epoch = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m_epoch = re.search(r'Train epoch\s+(\d+)', line)
            if m_epoch:
                current_epoch = int(m_epoch.group(1))
            m_test = re.search(r'TEST:\s+PSNR_S:\s+([-\d\.]+)\s+\|\s+PSNR_C:\s+([-\d\.]+)', line)
            if m_test and current_epoch is not None:
                psnr_s = float(m_test.group(1))
                psnr_r = float(m_test.group(2))
                records.append({
                    'file': os.path.basename(path),
                    'bits': bits,
                    'calibration': calibration,
                    'epoch': current_epoch,
                    'PSNR_s': psnr_s,
                    'PSNR_r': psnr_r,
                })
    return records


def load_logs(log_dir: str = LOG_DIR) -> pd.DataFrame:
    records = []
    for log_path in glob.glob(os.path.join(log_dir, '*.log')):
        records.extend(parse_log(log_path))
    return pd.DataFrame(records)


def plot_metrics(df: pd.DataFrame) -> None:
    if df.empty:
        print('No log data parsed.')
        return
    bits_values = sorted(df['bits'].unique())
    fig, axes = plt.subplots(
        len(bits_values), 1, figsize=(8, 4 * len(bits_values)), sharex=False
    )
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax, bits in zip(axes, bits_values):
        subset_bits = df[df['bits'] == bits]
        if subset_bits.empty:
            continue
        for calib, sub in subset_bits.groupby('calibration'):
            calib_label = calib if calib is not None else 'unknown'
            label_prefix = f'{bits}-bit calib {calib_label}'
            ax.plot(sub['epoch'], sub['PSNR_s'], label=f'{label_prefix} PSNR_s')
            ax.plot(
                sub['epoch'],
                sub['PSNR_r'],
                label=f'{label_prefix} PSNR_r',
                linestyle='--',
            )
        ax.set_title(f'{bits}-bit Results')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('PSNR')
        if ax.lines:
            ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = load_logs()
    print(df.head())
    plot_metrics(df)


if __name__ == '__main__':
    main()
