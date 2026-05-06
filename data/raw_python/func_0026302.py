def display_image_file(fn, width='auto', height='auto', preserve_aspect_ratio=None):
    """
    Display an image in the terminal.

    A newline is not printed.

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
    with open(os.path.realpath(os.path.expanduser(fn)), 'rb') as f:
        sys.stdout.buffer.write(image_bytes(f.read(), filename=fn,
            width=width, height=height,
            preserve_aspect_ratio=preserve_aspect_ratio))