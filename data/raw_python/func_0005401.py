def image_mime_type(data):
    """Return the MIME type of the image data (a bytestring).
    """
    # This checks for a jpeg file with only the magic bytes (unrecognized by
    # imghdr.what). imghdr.what returns none for that type of file, so
    # _wider_test_jpeg is run in that case. It still returns None if it didn't
    # match such a jpeg file.
    kind = _imghdr_what_wrapper(data)
    if kind in ['gif', 'jpeg', 'png', 'tiff', 'bmp']:
        return 'image/{0}'.format(kind)
    elif kind == 'pgm':
        return 'image/x-portable-graymap'
    elif kind == 'pbm':
        return 'image/x-portable-bitmap'
    elif kind == 'ppm':
        return 'image/x-portable-pixmap'
    elif kind == 'xbm':
        return 'image/x-xbitmap'
    else:
        return 'image/x-{0}'.format(kind)