import time

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataload.init import getdata
from eval import test
from log_report import LogReport
from log_report import TestReport
from utils.utils import checkpoint, gpu_manage, save_named_checkpoint

from .losses import CharbonnierLoss, EdgeLoss
from .model import MPRNet


def train(config):
    gpu_manage(config)

    print("===> Loading datasets")
    train_dataset, validation_dataset = getdata(config)
    print("train dataset:", len(train_dataset))
    print("validation dataset:", len(validation_dataset))

    training_data_loader = DataLoader(
        dataset=train_dataset,
        num_workers=config.threads,
        batch_size=config.batchsize,
        shuffle=True,
    )
    validation_data_loader = DataLoader(
        dataset=validation_dataset,
        num_workers=config.threads,
        batch_size=config.validation_batchsize,
        shuffle=False,
    )

    print("===> Loading models")
    gen = MPRNet(in_c=config.in_ch, out_c=config.in_ch)
    print("Total params: %.2fM" % (sum(p.numel() for p in gen.parameters()) / 1000000.0))

    if config.gen_init is not None:
        param = torch.load(config.gen_init, map_location=config.device)
        gen.load_state_dict(param)
        print("load {} as pretrained model".format(config.gen_init))

    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)

    gen = gen.to(config.device)
    criterion_char = CharbonnierLoss()
    criterion_edge = EdgeLoss()
    criterionMSE = nn.MSELoss()

    logreport = LogReport(log_dir=config.out_dir)
    validationreport = TestReport(log_dir=config.out_dir)

    print("===> begin")
    start_time = time.time()
    best_psnr = float("-inf")
    best_ssim = float("-inf")

    for epoch in range(1, config.epoch + 1):
        epoch_start_time = time.time()

        for _, batch in enumerate(training_data_loader, 1):
            real_a_cpu, real_b_cpu, m_cpu = batch[0], batch[1], batch[2]
            real_a = Variable(real_a_cpu).to(config.device)
            real_b = Variable(real_b_cpu).to(config.device)
            _ = Variable(m_cpu).to(config.device)

            opt_gen.zero_grad()

            restored = gen(real_a, isTrain=True)
            loss_char = sum(criterion_char(stage, real_b) for stage in restored)
            loss_edge = sum(criterion_edge(stage, real_b) for stage in restored)
            loss = loss_char + (0.05 * loss_edge)

            loss.backward()
            opt_gen.step()

        print("epoch", epoch, "finished, use time", time.time() - epoch_start_time)
        with torch.no_grad():
            log_validation = test(config, validation_data_loader, gen, criterionMSE, epoch)
            validationreport(log_validation)
        print("validation finished")

        current_psnr = log_validation.get("psnr", float("-inf"))
        current_ssim = log_validation.get("ssim", float("-inf"))

        if current_psnr > best_psnr:
            best_psnr = current_psnr
            save_named_checkpoint(config, "best_psnr.pth", gen)
            print("best psnr updated to {:.4f} at epoch {}".format(best_psnr, epoch))

        if current_ssim > best_ssim:
            best_ssim = current_ssim
            save_named_checkpoint(config, "best_ssim.pth", gen)
            print("best ssim updated to {:.4f} at epoch {}".format(best_ssim, epoch))

        if epoch % config.snapshot_interval == 0:
            checkpoint(config, epoch, gen)

        logreport.save_lossgraph()
        validationreport.save_lossgraph()

    print("training time:", time.time() - start_time)
