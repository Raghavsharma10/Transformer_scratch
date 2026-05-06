def imsave(filename, im, format=None):
    """Save image data to disk

    Requires imageio or PIL.

    Parameters
    ----------
    filename : str
        Filename to write.
    im : array
        Image data.
    format : str | None
        Format of the file. If None, it will be inferred from the filename.

    See also
    --------
    imread, read_png, write_png
    """
    # Import imageio or PIL
    imageio, PIL = _check_img_lib()
    if imageio is not None:
        return imageio.imsave(filename, im, format)
    elif PIL is not None:
        pim = PIL.Image.fromarray(im)
        pim.save(filename, format)
    else:
        raise RuntimeError("imsave requires the imageio or PIL package.")