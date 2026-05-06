def hexdump_iter( source, start=None, end=None, length=None, major_len=8, minor_len=4, colour=True, address_base=None ):
    """Return the contents of a byte string in tabular hexadecimal/ASCII format.
    
    source
        The byte string to print.

    start
        Start offset to read from (default: start)

    end
        End offset to stop reading at (default: end)

    length
        Length to read in (optional replacement for end)

    major_len
        Number of hexadecimal groups per line

    minor_len
        Number of bytes per hexadecimal group

    colour
        Add ANSI colour formatting to output (default: true)

    address_base
        Base address to use for labels (default: start)

    Raises ValueError if both end and length are defined.
    """
    assert is_bytes( source )
    start, end = bounds( start, end, length, len( source ) )

    start = max( start, 0 )
    end = min( end, len( source ) )
    if len( source ) == 0 or (start == end == 0):
        return
    address_base_offset = address_base-start if address_base is not None else 0

    for offset in range( start, end, minor_len*major_len ):
        yield ansi.format_hexdump_line( source, offset, end, major_len, minor_len, colour, address_base_offset=address_base_offset )
    return