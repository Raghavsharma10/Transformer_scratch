def image_bytes(b, filename=None, inline=1, width='auto', height='auto',
    preserve_aspect_ratio=None):
    """
    Return a bytes string that displays image given by bytes b in the terminal

    If filename=None, the filename defaults to "Unnamed file"

    width and height are strings, following the format

        N: N character cells.

        Npx: N pixels.

        N%: N percent of the session's width or height.

        'auto': The image's inherent size will be used to determine an appropriate
                dimension.

    preserve_aspect_ratio sets whether the aspect ratio of the image is
    preserved. The default (None) is True unless both width and height are
    set.

    See https://www.iterm2.com/documentation-images.html
    """
    if preserve_aspect_ratio is None:
        if width != 'auto' and height != 'auto':
            preserve_aspect_ratio = False
        else:
            preserve_aspect_ratio = True
    data = {
        'name': base64.b64encode((filename or 'Unnamed file').encode('utf-8')).decode('ascii'),
        'inline': inline,
        'size': len(b),
        'base64_img': base64.b64encode(b).decode('ascii'),
        'width': width,
        'height': height,
        'preserve_aspect_ratio': int(preserve_aspect_ratio),
        }
    # IMAGE_CODE is a string because bytes doesn't support formatting
    return IMAGE_CODE.format(**data).encode('ascii')