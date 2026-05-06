def hexdump(src, length=16, sep='.'):
    """
    Returns src in hex dump.
    From https://gist.github.com/ImmortalPC/c340564823f283fe530b

    :param length: Nb Bytes by row.
    :param sep: For the text part, sep will be used for non ASCII char.
    :return: The hexdump
    """
    result = []

    for i in range(0, len(src), length):
        sub_src = src[i:i + length]
        hexa = ''
        for h in range(0, len(sub_src)):
            if h == length / 2:
                hexa += ' '
            h = sub_src[h]
            if not isinstance(h, int):
                h = ord(h)
            h = hex(h).replace('0x', '')
            if len(h) == 1:
                h = '0' + h
            hexa += h + ' '

        hexa = hexa.strip(' ')
        text = ''
        for c in sub_src:
            if not isinstance(c, int):
                c = ord(c)
            if 0x20 <= c < 0x7F:
                text += chr(c)
            else:
                text += sep
        result.append(('%08X:  %-' + str(length * (2 + 1) + 1) + 's  |%s|')
                      % (i, hexa, text))

    return '\n'.join(result)