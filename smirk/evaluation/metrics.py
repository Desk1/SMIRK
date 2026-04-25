import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from smirk.utils.image import normalize, crop_and_resize


def ASR(predictions, target):
    prediction = predictions.argmax(dim=1).item()

    return (prediction == target)

def SSIM(gen_image, target_image):
    ssim = StructuralSimilarityIndexMeasure()

    return ssim(gen_image, target_image)

def PSNR(gen_image, target_image):
    psnr = PeakSignalNoiseRatio()

    return psnr(gen_image, target_image)