import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 파일명과 레이블을 리스트로 준비
files = [
    ("./logging/train__211224-105010.log", "FP32"),
    ("./logging/train__8bit_ep50_calib20.log", "8bit"),
    ("./logging/train__4bit_ep50_calib20.log", "4bit"),
]

def extract_psnr_loss(filename):
    psnr_r, psnr_s = [], []
    total_loss = None
    psnr_pattern = re.compile(r"PSNR_S:\s*([-\d\.]+)\s*\|\s*PSNR_C:\s*([-\d\.]+)")
    loss_pattern = re.compile(r"Train epoch \d+:\s+Loss: ([\d\.]+)")
    with open(filename, encoding="utf-8") as f:
        for line in f:
            loss_match = loss_pattern.search(line)
            if loss_match:
                total_loss = float(loss_match.group(1))
            psnr_match = psnr_pattern.search(line)
            if psnr_match:
                psnr_s.append(float(psnr_match.group(1)))
                psnr_r.append(float(psnr_match.group(2)))
    return psnr_r, psnr_s, total_loss

# 원하는 x축 순서
order = ["FP32", "4bit", "8bit"]

summary = {}
for filename, label in files:
    psnr_r, psnr_s, final_total_loss = extract_psnr_loss(filename)
    final_psnr_r = psnr_r[-1] if psnr_r else 0
    final_psnr_s = psnr_s[-1] if psnr_s else 0
    if final_psnr_r == 0 or final_psnr_s == 0:
        print(f"경고: {label}에서 PSNR_S/PSNR_C 값을 찾지 못했습니다.")
    summary[label] = {
        "Model": label,
        "Final_PSNR_R": final_psnr_r,
        "Final_PSNR_S": final_psnr_s,
        "Final_Total_Loss": final_total_loss
    }

# 순서에 맞게 리스트 재정렬
model_names = []
psnr_r_list = []
psnr_s_list = []
loss_list = []
summary_list = []

for label in order:
    model_names.append(label)
    psnr_r_list.append(summary[label]["Final_PSNR_R"])
    psnr_s_list.append(summary[label]["Final_PSNR_S"])
    loss_list.append(summary[label]["Final_Total_Loss"])
    summary_list.append(summary[label])

# Bar chart
x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, psnr_r_list, width, label='PSNR_C', color='#1f77b4')
rects2 = ax.bar(x + width/2, psnr_s_list, width, label='PSNR_S', color='#ff7f0e')

ax.set_ylabel('PSNR (dB)')
ax.set_title('FP32 / 4bit / 8bit QAT Experiments')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("./logging/psnr_bar_comparison.png")
plt.show()

# CSV 저장
df = pd.DataFrame(summary_list)
df.to_csv("./loggingfinal_psnr_loss_summary.csv", index=False)
print(df)