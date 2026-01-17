from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import cv2
import torch
import numpy as np
class ESRGAN:
    def __init__(self, use_x2=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if use_x2:
            model_path = 'RealESRGAN_x2plus.pth' #https://huggingface.co/dtarnow/UPscaler/blob/main/RealESRGAN_x2plus.pth
            scale = 2
        else:
            model_path = 'RealESRGAN_x4plus.pth' #https://huggingface.co/lllyasviel/Annotators/blob/main/RealESRGAN_x4plus.pth
            scale = 4
        self.scale = scale
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale
        )
        self.upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            device=self.device
        )

    def upscale(self, img, tile_size=512, tile_pad=10):
        """
        img : numpy array BGR
        tile_size : tamanho do bloco (tile)
        tile_pad : padding extra para suavizar bordas
        """
        h, w, c = img.shape
        scale = self.scale

        # Criar imagem de saída maior
        out_img = np.zeros((h*scale, w*scale, 3), dtype=np.uint8)

        # Processar tiles
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # Calcular região com padding
                y1 = max(y - tile_pad, 0)
                y2 = min(y + tile_size + tile_pad, h)
                x1 = max(x - tile_pad, 0)
                x2 = min(x + tile_size + tile_pad, w)

                tile = img[y1:y2, x1:x2]

                # Upscale do tile
                out_tile, _ = self.upsampler.enhance(tile)

                # Remover padding do tile upscalado
                pad_top = (y - y1) * scale
                pad_left = (x - x1) * scale
                pad_bottom = pad_top + (min(tile_size, h - y)) * scale
                pad_right = pad_left + (min(tile_size, w - x)) * scale

                out_img[y*scale:(y+min(tile_size,h-y))*scale,
                        x*scale:(x+min(tile_size,w-x))*scale] = out_tile[pad_top:pad_bottom, pad_left:pad_right]

        torch.cuda.empty_cache()
        return out_img
