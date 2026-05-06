def is_zhuyin_compatible(s):
    """Checks if *s* is consists of Zhuyin-compatible characters.

    This does not check if *s* contains valid Zhuyin syllables; for that
    see :func:`is_zhuyin`.

    Besides Zhuyin characters and tone marks, spaces are also accepted.
    This function checks that all characters in *s* exist in
    :data:`zhon.zhuyin.characters`, :data:`zhon.zhuyin.marks`, or ``' '``.

    """
    printable_zhuyin = zhon.zhuyin.characters + zhon.zhuyin.marks + ' '
    return _is_pattern_match('[%s]+' % printable_zhuyin, s)