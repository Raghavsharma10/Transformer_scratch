def hexdump( source, start=None, end=None, length=None, major_len=8, minor_len=4, colour=True, address_base=None ):
    """Print the contents of a byte string in tabular hexadecimal/ASCII format.
    
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
    for line in hexdump_iter( source, start, end, length, major_len, minor_len, colour, address_base ):
        print( line )