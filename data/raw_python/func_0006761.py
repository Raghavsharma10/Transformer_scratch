def is_light(syll):
    '''Return True if 'syll' is light.'''
    return re.match(r'(^|[^ieaouäöy]+)[ieaouäöy]{1}$', syll, flags=FLAGS)