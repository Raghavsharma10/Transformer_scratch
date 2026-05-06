def hexdump_diff( source1, source2, start=None, end=None, length=None, major_len=8, minor_len=4, colour=True, before=2, after=2, address_base=None ):
    """Returns the differences between two byte strings in tabular hexadecimal/ASCII format.

    source1
        The first byte string source.

    source2
        The second byte string source.

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

    before
        Number of lines of context preceeding a match to show

    after
        Number of lines of context following a match to show

    address_base
        Base address to use for labels (default: start)

    Raises ValueError if both end and length are defined.
    """
    for line in hexdump_diff_iter( source1, source2, start, end, length, major_len, minor_len, colour, before, after, address_base ):
        print( line )