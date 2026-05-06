def display_image_bytes(b, filename=None, inline=1, width='auto',
    height='auto', preserve_aspect_ratio=None):
    """
    Display the image given by the bytes b in the terminal.

    If filename=None the filename defaults to "Unnamed file".

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
    sys.stdout.buffer.write(image_bytes(b, filename=filename, inline=inline,
        width=width, height=height, preserve_aspect_ratio=preserve_aspect_ratio))
    sys.stdout.write('\n')