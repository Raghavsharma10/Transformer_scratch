def is_valid_sound_tuple(sound_tuple, final_form=True):
    """
    Check if a character combination complies to Vietnamese phonology.
    The basic idea is that if one can pronunce a sound_tuple then it's valid.
    Sound tuples containing consonants exclusively (almost always
    abbreviations) are also valid.

    Input:
        sound_tuple - a SoundTuple
        final_form  - whether the tuple represents a complete word
    Output:
        True if the tuple seems to be Vietnamese, False otherwise.
    """

    # We only work with lower case
    sound_tuple = SoundTuple._make([s.lower() for s in sound_tuple])

    # Words with no vowel are always valid
    # FIXME: This looks like it should be toggled by a config key.
    if not sound_tuple.vowel:
        result = True
    elif final_form:
        result = \
            has_valid_consonants(sound_tuple) and \
            has_valid_vowel(sound_tuple) and \
            has_valid_accent(sound_tuple)
    else:
        result = \
            has_valid_consonants(sound_tuple) and \
            has_valid_vowel_non_final(sound_tuple)

    return result