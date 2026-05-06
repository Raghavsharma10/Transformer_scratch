def same_syllabic_feature(ch1, ch2):
    '''Return True if ch1 and ch2 are both vowels or both consonants.'''
    if ch1 == '.' or ch2 == '.':
        return False

    ch1 = 'V' if ch1 in VOWELS else 'C' if ch1 in CONSONANTS else None
    ch2 = 'V' if ch2 in VOWELS else 'C' if ch2 in CONSONANTS else None

    return ch1 == ch2