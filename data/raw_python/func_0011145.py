def peel_off_esc_code(s):
    r"""Returns processed text, the next token, and unprocessed text

    >>> front, d, rest = peel_off_esc_code('some[2Astuff')
    >>> front, rest
    ('some', 'stuff')
    >>> d == {'numbers': [2], 'command': 'A', 'intermed': '', 'private': '', 'csi': '\x1b[', 'seq': '\x1b[2A'}
    True
    """
    p = r"""(?P<front>.*?)
            (?P<seq>
                (?P<csi>
                    (?:[]\[)
                    |
                    ["""+'\x9b' + r"""])
                (?P<private>)
                (?P<numbers>
                    (?:\d+;)*
                    (?:\d+)?)
                (?P<intermed>""" + '[\x20-\x2f]*)' + r"""
                (?P<command>""" + '[\x40-\x7e]))' + r"""
            (?P<rest>.*)"""
    m1 = re.match(p, s, re.VERBOSE)  # multibyte esc seq
    m2 = re.match('(?P<front>.*?)(?P<seq>(?P<csi>)(?P<command>[\x40-\x5f]))(?P<rest>.*)', s)  # 2 byte escape sequence
    if m1 and m2:
        m = m1 if len(m1.groupdict()['front']) <= len(m2.groupdict()['front']) else m2
        # choose the match which has less processed text in order to get the
        # first escape sequence
    elif m1: m = m1
    elif m2: m = m2
    else: m = None

    if m:
        d = m.groupdict()
        del d['front']
        del d['rest']
        if 'numbers' in d and all(d['numbers'].split(';')):
            d['numbers'] = [int(x) for x in d['numbers'].split(';')]

        return m.groupdict()['front'], d, m.groupdict()['rest']
    else:
        return s, None, ''