def get_vowel(syll):
    '''Return the firstmost vowel in 'syll'.'''
    return re.search(r'([ieaouäöy]{1})', syll, flags=FLAGS).group(1).upper()