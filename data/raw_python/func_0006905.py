def nonalpha_split(string):
    '''Split 'string' along any punctuation or whitespace.'''
    return re.findall(r'[%s]+|[^%s]+' % (A, A), string, flags=FLAGS)