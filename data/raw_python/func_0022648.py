def hexdump(src, length=16, sep='.'):
    """
    Hexdump function by sbz and 7h3rAm on Github:
    (https://gist.github.com/7h3rAm/5603718).
    :param src: Source, the string to be shown in hexadecimal format
    :param length: Number of hex characters to print in one row
    :param sep: Unprintable characters representation
    :return:
    """
    filtr = ''.join([(len(repr(chr(x))) == 3) and chr(x) or sep for x in range(256)])
    lines = []
    for c in xrange(0, len(src), length):
        chars = src[c:c+length]
        hexstring = ' '.join(["%02x" % ord(x) for x in chars])
        if len(hexstring) > 24:
            hexstring = "%s %s" % (hexstring[:24], hexstring[24:])
        printable = ''.join(["%s" % ((ord(x) <= 127 and filtr[ord(x)]) or sep) for x in chars])
        lines.append("     %02x:  %-*s  |%s|\n" % (c, length*3, hexstring, printable))
    print(''.join(lines))