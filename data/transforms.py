from jittor import transform

def get_transform(new_size=None):
    """
    obtain the image transforms required for the input data
    :param new_size: size of the resized images
    :return: image_transform => transform object from TorchVision
    """
    if new_size is not None:
        image_transform = transform.Compose([
            transform.RandomHorizontalFlip(),
            transform.Resize(new_size),
            transform.ToTensor(),
            transform.ImageNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    else:
        image_transform = transform.Compose([
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.ImageNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    return image_transform