def pixdump_iter( source, start=None, end=None, length=None, width=64, height=None, palette=None ):
    """Return the contents of a byte string as a 256 colour image.

    source
        The byte string to print.

    start
        Start offset to read from (default: start)

    end
        End offset to stop reading at (default: end)

    length
        Length to read in (optional replacement for end)

    width
        Width of image to render in pixels (default: 64)

    height
        Height of image to render in pixels (default: auto)

    palette
        List of Colours to use (default: test palette)
    """
    assert is_bytes( source )

    if not palette:
        palette = colour.TEST_PALETTE

    start = 0 if (start is None) else start
    if (end is not None) and (length is not None):
        raise ValueError( 'Can\'t define both an end and a length!' )
    elif (length is not None):
        end = start+length
    elif (end is not None):
        pass
    else:
        end = len( source )

    start = max( start, 0 )
    end = min( end, len( source ) )
    if len( source ) == 0 or (start == end == 0):
        return iter(())

    if height is None:
        height = math.ceil( (end-start)/width )

    def data_fetch( x_pos, y_pos, frame ):
        index = y_pos*width + x_pos + start
        if index >= end:
            return (0, 0, 0, 0)
        return palette[source[index]]

    return ansi.format_image_iter( data_fetch, width=width, height=height )