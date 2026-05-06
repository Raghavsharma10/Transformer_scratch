def imread(filename, format=None):
    """Read image data from disk

    Requires imageio or PIL.

    Parameters
    ----------
    filename : str
        Filename to read.
    format : str | None
        Format of the file. If None, it will be inferred from the filename.

    Returns
    -------
    data : array
        Image data.

    See also
    --------
    imsave, read_png, write_png
    """
    imageio, PIL = _check_img_lib()
    if imageio is not None:
        return imageio.imread(filename, format)
    elif PIL is not None:
        im = PIL.Image.open(filename)
        if im.mode == 'P':
            im = im.convert()
        # Make numpy array
        a = np.asarray(im)
        if len(a.shape) == 0:
            raise MemoryError("Too little memory to convert PIL image to "
                              "array")
        return a
    else:
        raise RuntimeError("imread requires the imageio or PIL package.")