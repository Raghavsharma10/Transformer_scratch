def decompose_hangul_syllable(hangul_syllable, fully_decompose=False):
    """
    Function for taking a Unicode scalar value representing a Hangul syllable and decomposing it into a tuple
    representing the scalar values of the decomposed (canonical decomposition) Jamo.  If the Unicode scalar value
    passed in is not in the range of Hangul syllable values (as defined in UnicodeData.txt), a ValueError will be
    raised.

    The algorithm for doing the decomposition is described in the Unicode Standard, ch. 03, section 3.12,
    "Conjoining Jamo Behavior".

    Example: U+D4DB -> (U+D4CC, U+11B6)  # (canonical decomposition, default)
             U+D4DB -> (U+1111, U+1171, U+11B6)  # (full canonical decomposition)

    :param hangul_syllable: Unicode scalar value for Hangul syllable
    :param fully_decompose: Boolean indicating whether or not to do a canonical decomposition (default behavior is
                            fully_decompose=False) or a full canonical decomposition (fully_decompose=True)
    :return: Tuple of Unicode scalar values for the decomposed Jamo.
    """
    if not _is_hangul_syllable(hangul_syllable):
        raise ValueError("Value passed in does not represent a Hangul syllable!")
    s_index = hangul_syllable - S_BASE

    if fully_decompose:
        l_index = s_index // N_COUNT
        v_index = (s_index % N_COUNT) // T_COUNT
        t_index = s_index % T_COUNT
        l_part = L_BASE + l_index
        v_part = V_BASE + v_index
        t_part = (T_BASE + t_index) if t_index > 0 else None
        return l_part, v_part, t_part
    else:
        if _get_hangul_syllable_type(hangul_syllable) == "LV":  # Hangul_Syllable_Type = LV
            l_index = s_index // N_COUNT
            v_index = (s_index % N_COUNT) // T_COUNT
            l_part = L_BASE + l_index
            v_part = V_BASE + v_index
            return l_part, v_part
        else:  # Assume Hangul_Syllable_Type = LVT
            lv_index = (s_index // T_COUNT) * T_COUNT
            t_index = s_index % T_COUNT
            lv_part = S_BASE + lv_index
            t_part = T_BASE + t_index
            return lv_part, t_part