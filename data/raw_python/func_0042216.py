def scale_and_crop(im, crop_spec):
    """
    Scale and Crop.
    """
    im = im.crop((crop_spec.x, crop_spec.y, crop_spec.x2, crop_spec.y2))

    if crop_spec.width and crop_spec.height:
        im = im.resize((crop_spec.width, crop_spec.height),
                   resample=Image.ANTIALIAS)

    return im