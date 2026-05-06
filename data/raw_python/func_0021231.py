def _hangul_char_to_jamo(syllable):
    """Return a 3-tuple of lead, vowel, and tail jamo characters.
    Note: Non-Hangul characters are echoed back.
    """
    if is_hangul_char(syllable):
        rem = ord(syllable) - _JAMO_OFFSET
        tail = rem % 28
        vowel = 1 + ((rem - tail) % 588) // 28
        lead = 1 + rem // 588
        if tail:
            return (chr(lead + _JAMO_LEAD_OFFSET),
                    chr(vowel + _JAMO_VOWEL_OFFSET),
                    chr(tail + _JAMO_TAIL_OFFSET))
        else:
            return (chr(lead + _JAMO_LEAD_OFFSET),
                    chr(vowel + _JAMO_VOWEL_OFFSET))
    else:
        return syllable