def get_image_sizes(image):
    """Given an ImageField `image`, returns a list of images sizes in this
    form:

    [
        {
            "url": "http://example.com/xxx.jpg",
            "width": 1440,
            "height": 960
        },
        [...]
    ]"""

    # It is possible to have the same width appear more than once, if
    # THUMBNAIL_UPSCALE is set to False and the image's width is less than the
    # largest value in FLEXIBLE_IMAGE_SIZES. So keep track of widths and
    # don't output more than one image with the same width (which would result
    # in an invalid `srcset` attribute).
    sizes = []
    seen_widths = []

    for size in settings_sizes():
        img = get_thumbnail_shim(image, size)

        if img.width in seen_widths:
            continue

        seen_widths.append(img.width)

        sizes.append({
            "url": img.url,
            "width": img.width,
            "height": img.height,
        })
    return sizes