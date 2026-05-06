def syllable_split(string):
    '''Split 'string' into (stressed) syllables and punctuation/whitespace.'''
    p = r'\'[%s]+|`[%s]+|[%s]+|[^%s\'`\.]+|[^\.]{1}' % (A, A, A, A)
    return re.findall(p, string, flags=FLAGS)