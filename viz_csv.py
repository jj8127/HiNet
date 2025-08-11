import os
import pandas as pd
import matplotlib.pyplot as plt

# 시각화할 csv 파일 리스트
csv_files = [
    "./result_csv/pretrained_QAT_model.csv",
    #"./result_csv/finetuned_model.csv",
    #"./result_csv/finetuned_QAT_model.csv"
    "./result_csv/pretrained_model.csv"
]

# 데이터 로드 및 평균 제외
data_dict = {}
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df = df[df['img_name'] != 'average']
    model_name = os.path.splitext(os.path.basename(csv_file))[0]
    data_dict[model_name] = df

# 지표 리스트
metrics = [
    ("psnr_c", "PSNR_C"),
    ("psnr_r", "PSNR_R"),
    ("ssim_c", "SSIM_C"),
    ("ssim_r", "SSIM_R"),
    ("ssim_avg", "SSIM_AVG")
]

# subplot 5개 (1x5)
fig, axes = plt.subplots(1, len(metrics), figsize=(24, 5))

for idx, (metric, metric_label) in enumerate(metrics):
    ax = axes[idx]
    for model_name, df in data_dict.items():
        ax.plot(df[metric].astype(float).values, label=model_name)
    ax.set_title(metric_label)
    ax.set_xlabel("Image Index")
    ax.set_ylabel(metric_label)
    ax.grid(True, linestyle="--", alpha=0.6)
    if idx == 0:
        ax.legend()

plt.tight_layout()
plt.savefig("./result_csv/pretrain VS pretrain_QAT compare.png")
plt.show()
