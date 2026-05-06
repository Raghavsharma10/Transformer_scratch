def compose_hangul_syllable(jamo):
    """
    Function for taking a tuple or list of Unicode scalar values representing Jamo and composing it into a Hangul
    syllable.  If the values in the list or tuple passed in are not in the ranges of Jamo, a ValueError will be raised.

    The algorithm for doing the composition is described in the Unicode Standard, ch. 03, section 3.12, "Conjoining Jamo
    Behavior."

    Example: (U+1111, U+1171) -> U+D4CC
             (U+D4CC, U+11B6) -> U+D4DB
             (U+1111, U+1171, U+11B6) -> U+D4DB

    :param jamo: Tuple of list of Jamo to compose
    :return: Composed Hangul syllable
    """
    fmt_str_invalid_sequence = "{0} does not represent a valid sequence of Jamo!"
    if len(jamo) == 3:
        l_part, v_part, t_part = jamo
        if not (l_part in range(0x1100, 0x1112 + 1) and
                v_part in range(0x1161, 0x1175 + 1) and
                t_part in range(0x11a8, 0x11c2 + 1)):
            raise ValueError(fmt_str_invalid_sequence.format(jamo))
        l_index = l_part - L_BASE
        v_index = v_part - V_BASE
        t_index = t_part - T_BASE
        lv_index = l_index * N_COUNT + v_index * T_COUNT
        return S_BASE + lv_index + t_index
    elif len(jamo) == 2:
        if jamo[0] in range(0x1100, 0x1112 + 1) and jamo[1] in range(0x1161, 0x1175 + 1):
            l_part, v_part = jamo
            l_index = l_part - L_BASE
            v_index = v_part - V_BASE
            lv_index = l_index * N_COUNT + v_index * T_COUNT
            return S_BASE + lv_index
        elif _get_hangul_syllable_type(jamo[0]) == "LV" and jamo[1] in range(0x11a8, 0x11c2 + 1):
            lv_part, t_part = jamo
            t_index = t_part - T_BASE
            return lv_part + t_index
        else:
            raise ValueError(fmt_str_invalid_sequence.format(jamo))
    else:
        raise ValueError(fmt_str_invalid_sequence.format(jamo))