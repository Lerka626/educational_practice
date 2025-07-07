import random
from PIL import Image, ImageEnhance, ImageFilter
from torchvision.transforms import functional as F


class RandomBlur:
    """
    Случайное размытие (GaussianBlur или обычное Blur).
    """
    def __init__(self, p=0.5, radius_range=(1, 3)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img


class RandomPerspectiveCustom:
    """
    Случайная перспектива (с изменяемой степенью и вероятностью).
    """
    def __init__(self, distortion_scale=0.5, p=0.5):
        self.distortion_scale = distortion_scale
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            width, height = img.size
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            return F.perspective(img, startpoints, endpoints, interpolation=Image.BICUBIC, fill=0)
        return img

    @staticmethod
    def get_params(width, height, distortion_scale):
        half_width = width // 2
        half_height = height // 2
        topleft = [random.randint(0, int(distortion_scale * half_width)),
                   random.randint(0, int(distortion_scale * half_height))]
        topright = [random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(0, int(distortion_scale * half_height))]
        botright = [random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(height - int(distortion_scale * half_height) - 1, height - 1)]
        botleft = [random.randint(0, int(distortion_scale * half_width)),
                   random.randint(height - int(distortion_scale * half_height) - 1, height - 1)]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints


class RandomBrightnessContrast:
    """
    Случайная яркость и контрастность.
    """
    def __init__(self, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3), p=0.5):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            # Яркость
            brightness_factor = random.uniform(*self.brightness_range)
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)
            # Контрастность
            contrast_factor = random.uniform(*self.contrast_range)
            img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        return img
