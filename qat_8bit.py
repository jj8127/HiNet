#!/usr/bin/env python
import torch
import torch.nn
import torch.optim
import math
import numpy as np
from model import *
import config as c
from tensorboardX import SummaryWriter
import datasets
import viz
import modules.Unet_common as common
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()
    return noise

def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)

def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.to(device)

def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def computePSNR(origin, pred):
    origin = np.array(origin).astype(np.float32)
    pred = np.array(pred).astype(np.float32)
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)

def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict, strict=False)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except Exception:
        print('Cannot load optimizer for some reason or other')

def get_custom_save_name(
    base_name,
    qat="qat",
    bit=8,
    epoch=None,
    calib=None,
    ext=".pt"
):
    name = f"{base_name}_{qat}_{bit}bit"
    if epoch is not None:
        name += f"_ep{epoch}"
    if calib is not None:
        name += f"_calib{calib}"
    name += ext
    return name

#####################
# Model initialize: #
#####################
net = Model()
net.cuda()
init_model(net)

# 전체 파라미터 freeze
for param in net.parameters():
    param.requires_grad = False

# 부분 파라미터만 trainable로 설정 (예: 'inv'와 'conv' 이름 포함된 파라미터)
for name, param in net.named_parameters():
    if "inv" in name and "conv" in name:
        param.requires_grad = True
print("Trainable params after freezing and partial unfreeze:")
print(get_parameter_number(net))

net = torch.nn.DataParallel(net, device_ids=c.device_ids)
params_trainable = [p for p in net.parameters() if p.requires_grad]
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

# Load pretrained weights for finetuning
load(c.PRETRAINED_MODEL)

dwt = common.DWT()
iwt = common.IWT()

now = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"logs_{now}"
os.makedirs(save_dir, exist_ok=True)
log_csv_path = os.path.join(save_dir, 'training_log.csv')

train_losses = []
psnr_s_list = []
psnr_c_list = []
epochs = []

finetuned_name = os.path.splitext(os.path.basename(c.suffix))[0]
bit = 8  # QAT bit수, 상황에 따라 수정
calib_num = c.val_freq

try:
    writer = SummaryWriter(comment='finetune', filename_suffix='steg')

    for i_epoch in range(c.epochs):
        i_epoch = i_epoch + c.trained_epoch + 1
        loss_history = []

        for i_batch, (cover, secret) in enumerate(datasets.trainloader):
            cover = cover.to(device)
            secret = secret.to(device)
            cover_input = dwt(cover)
            secret_input = dwt(secret)
            input_img = torch.cat((cover_input, secret_input), 1)
            output = net(input_img)
            output_steg = output.narrow(1, 0, 4 * c.channels_in)
            output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
            steg_img = iwt(output_steg)
            output_z_guass = gauss_noise(output_z.shape)
            output_rev = torch.cat((output_steg, output_z_guass), 1)
            output_image = net(output_rev, rev=True)
            secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
            secret_rev = iwt(secret_rev)
            g_loss = guide_loss(steg_img.cuda(), cover.cuda())
            r_loss = reconstruction_loss(secret_rev, secret)
            steg_low = output_steg.narrow(1, 0, c.channels_in)
            cover_low = cover_input.narrow(1, 0, c.channels_in)
            l_loss = low_frequency_loss(steg_low, cover_low)
            total_loss = c.lamda_reconstruction * r_loss + c.lamda_guide * g_loss + c.lamda_low_frequency * l_loss
            total_loss.backward()
            optim.step()
            optim.zero_grad()
            loss_history.append([total_loss.item(), 0.])

        epoch_losses = np.mean(np.array(loss_history), axis=0)
        epoch_losses[1] = np.log10(optim.param_groups[0]['lr'])

        # 검증(calibration)
        if i_epoch % c.val_freq == 0:
            with torch.no_grad():
                psnr_s = []
                psnr_c = []
                net.eval()
                for cover, secret in datasets.testloader:
                    cover = cover.to(device)
                    secret = secret.to(device)
                    cover_input = dwt(cover)
                    secret_input = dwt(secret)
                    input_img = torch.cat((cover_input, secret_input), 1)
                    output = net(input_img)
                    output_steg = output.narrow(1, 0, 4 * c.channels_in)
                    steg = iwt(output_steg)
                    output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
                    output_z = gauss_noise(output_z.shape)
                    output_steg = output_steg.cuda()
                    output_rev = torch.cat((output_steg, output_z), 1)
                    output_image = net(output_rev, rev=True)
                    secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
                    secret_rev = iwt(secret_rev)
                    secret_rev = secret_rev.cpu().numpy().squeeze() * 255
                    np.clip(secret_rev, 0, 255)
                    secret = secret.cpu().numpy().squeeze() * 255
                    np.clip(secret, 0, 255)
                    cover = cover.cpu().numpy().squeeze() * 255
                    np.clip(cover, 0, 255)
                    steg = steg.cpu().numpy().squeeze() * 255
                    np.clip(steg, 0, 255)
                    psnr_temp = computePSNR(secret_rev, secret)
                    psnr_s.append(psnr_temp)
                    psnr_temp_c = computePSNR(cover, steg)
                    psnr_c.append(psnr_temp_c)
                writer.add_scalars('PSNR_S', {'average psnr': np.mean(psnr_s)}, i_epoch)
                writer.add_scalars('PSNR_C', {'average psnr': np.mean(psnr_c)}, i_epoch)
        else:
            if len(psnr_s_list) > 0 and len(psnr_c_list) > 0:
                psnr_s = [psnr_s_list[-1]]
                psnr_c = [psnr_c_list[-1]]
            else:
                psnr_s = [0]
                psnr_c = [0]

        train_losses.append(epoch_losses[0])
        psnr_s_list.append(np.mean(psnr_s))
        psnr_c_list.append(np.mean(psnr_c))
        epochs.append(i_epoch)
        viz.show_loss(epoch_losses)
        writer.add_scalars('Train', {'Train_Loss': epoch_losses[0]}, i_epoch)

        # 중간 저장
        df = pd.DataFrame({
            'epoch': epochs,
            'train_loss': train_losses,
            'psnr_s': psnr_s_list,
            'psnr_c': psnr_c_list
        })
        df.to_csv(log_csv_path, index=False)

        # QAT 모델 저장 (주기별)
        if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
            save_file = get_custom_save_name(
                base_name=finetuned_name,
                qat="qat",
                bit=bit,
                epoch=i_epoch,
                calib=calib_num,
                ext=".pt"
            )
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, os.path.join(save_dir, save_file))
        weight_scheduler.step()

    # 최종 QAT 모델 저장
    final_save_file = get_custom_save_name(
        base_name=finetuned_name,
        qat="qat",
        bit=bit,
        epoch="final",
        calib=calib_num,
        ext=".pt"
    )
    torch.save({'opt': optim.state_dict(),
                'net': net.state_dict()}, os.path.join(save_dir, final_save_file))
    writer.close()

    # 그래프 이미지 저장
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, psnr_s_list, label='PSNR_S', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR_S')
    plt.title('PSNR_S Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'psnr_s_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, psnr_c_list, label='PSNR_C', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR_C')
    plt.title('PSNR_C Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'psnr_c_curve.png'))
    plt.close()

except Exception:
    if c.checkpoint_on_error:
        # 에러 상황도 이름에 'ABORT'로 구분
        abort_file = get_custom_save_name(
            base_name=finetuned_name,
            qat="qat",
            bit=bit,
            epoch="ABORT",
            calib=calib_num,
            ext=".pt"
        )
        torch.save({'opt': optim.state_dict(),
                    'net': net.state_dict()}, os.path.join(save_dir, abort_file))
    raise

finally:
    viz.signal_stop()
