def set_seed(seed = None):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        

def add_random_background(image, bg_color=None):
    bg_color = np.random.rand() * 255 if bg_color is None else bg_color
    image = np.array(image)
    rgb, alpha = image[..., :3], image[..., 3:]
    alpha = alpha.astype(np.float32) / 255.0
    image_new = rgb * alpha + bg_color * (1 - alpha)
    return Image.fromarray(image_new.astype(np.uint8))