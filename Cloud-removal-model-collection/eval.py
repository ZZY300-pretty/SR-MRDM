import numpy as np
from skimage.metrics import structural_similarity as SSIM
from torch.autograd import Variable
from utils.utils import save_image,save_imagenir
import torch
from skimage.metrics import peak_signal_noise_ratio as PSNR

try:
    import lpips
except ImportError:
    lpips = None

loss_fn = None


def get_lpips_model(device):
    global loss_fn
    if lpips is None:
        return None
    if loss_fn is None:
        loss_fn = lpips.LPIPS(net='alex', version=0.1)
    return loss_fn.to(device)

def caculate_lpips(img0, img1, device):
    lpips_model = get_lpips_model(device)
    if lpips_model is None:
        return None

    im1 = img0.float().unsqueeze(0).to(device) / 127.5 - 1.0
    im2 = img1.float().unsqueeze(0).to(device) / 127.5 - 1.0
    current_lpips_distance = lpips_model(im1, im2)
    return float(current_lpips_distance.detach().cpu().item())

def caculate_ssim(imgA, imgB):
    imgA1 = imgA.cpu().numpy().transpose(1, 2, 0)
    imgB1 = imgB.cpu().numpy().transpose(1, 2, 0)
    if imgA1.shape[2] == 3:
        gray_a = np.tensordot(imgA1, [0.298912, 0.586611, 0.114478], axes=1)
        gray_b = np.tensordot(imgB1, [0.298912, 0.586611, 0.114478], axes=1)
        return SSIM(gray_a, gray_b, data_range=255)

    score = 0.0
    for channel in range(imgA1.shape[2]):
        score += SSIM(imgA1[:, :, channel], imgB1[:, :, channel], data_range=255)
    return score / imgA1.shape[2]

def caculate_psnr( imgA, imgB):
    imgA1 = imgA.cpu().numpy().transpose(1, 2, 0)
    imgB1 = imgB.cpu().numpy().transpose(1, 2, 0)
    psnr = PSNR(imgA1, imgB1, data_range=255)
    return psnr

def get_image_arr(dataset):  #the id of the image you want to save 
    if(dataset=='RICE1'):
        return ['0','105','143','368','425','458','495']
    elif(dataset=='RICE2'):
        return ['49','17','185','209','309','619','408','630']
    elif(dataset=='T-Cloud'):
        return ['278','142','162','449','930','1261','1652']
    elif(dataset=='My14' or dataset=='My24'):
        return ['4','5','7','8','36','37','65']
    else:
        return []


def test(config, test_data_loader, gen, criterionMSE, epoch):
    device = getattr(config, "device", torch.device("cuda:0" if config.cuda else "cpu"))
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    avg_mse = 0
    lpips_count = 0
    for i, batch in enumerate(test_data_loader):
        x, t, filename = Variable(batch[0]), Variable(batch[1]),batch[3]
        x = x.to(device)
        t = t.to(device)

        prediction = gen(x)
        if isinstance(prediction, (tuple, list)):
            out = prediction[-1]
        else:
            out = prediction
        if epoch % config.snapshot_interval == 0 and epoch > 20 and filename[0] in get_image_arr(config.datasets_dir):
            if(x.shape[1]==3):
                save_image(config.out_dir, x, i, epoch, filename=filename[0]+'Cloudy')
                save_image(config.out_dir, t, i, epoch, filename=filename[0]+'GT')
                save_image(config.out_dir, out, i, epoch, filename=filename[0]+'CR')
            else:  #it handle the multispectral nir layer (the situation that image contain 4 band RGB and nir)
                save_imagenir(config.out_dir, x, i, epoch, filename=filename[0]+'Cloudy')
                save_imagenir(config.out_dir, t, i, epoch, filename=filename[0]+'GT')
                save_imagenir(config.out_dir, out, i, epoch, filename=filename[0]+'CR')

        imgA = (out[0]*255).clamp(0, 255).to(torch.uint8)
        imgB = (t[0]*255).clamp(0, 255).to(torch.uint8)
        mse = float(criterionMSE(out, t).item())
        psnr = caculate_psnr(imgA, imgB)
        c,w,h=imgA.shape
        if(imgA.shape[0]==4):
            lpips_value = 0.0
            ssim=0.0
            valid_lpips = 0
            for channel in range(imgA.shape[0]):
                imA = imgA[channel]
                imA = imA.expand(3,w,h)
                imB = imgB[channel]
                imB = imB.expand(3,w,h)
                ssim1 = caculate_ssim(imA, imB)
                lpips1 = caculate_lpips(imA, imB, device)
                ssim+=ssim1
                if lpips1 is not None:
                    lpips_value += lpips1
                    valid_lpips += 1
            ssim=ssim/imgA.shape[0]
            if valid_lpips > 0:
                lpips_value = lpips_value / valid_lpips
            else:
                lpips_value = None
        else:
            ssim = caculate_ssim(imgA, imgB)
            lpips_value = caculate_lpips(imgA, imgB, device)
        avg_psnr += psnr
        avg_ssim += ssim
        avg_mse += mse
        if lpips_value is not None:
            avg_lpips += lpips_value
            lpips_count += 1
    avg_mse = avg_mse / len(test_data_loader)
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)
    avg_lpips = avg_lpips / lpips_count if lpips_count else None

    print("===> Avg. MSE: {:.6f}".format(avg_mse))
    print("===> Avg. PSNR: {:.3f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f} dB".format(avg_ssim))
    if avg_lpips is None:
        print("===> Avg. LPIPS: skipped (lpips package not installed)")
    else:
        print("===> Avg. LPIPS: {:.4f}".format(avg_lpips))
    
    log_test = {}
    log_test['epoch'] = epoch
    log_test['mse'] = avg_mse
    log_test['psnr'] = avg_psnr
    log_test['ssim'] = avg_ssim
    if avg_lpips is not None:
        log_test['lpips'] = avg_lpips

    return log_test
