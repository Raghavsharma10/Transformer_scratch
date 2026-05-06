def filter_short(terms):
    '''
    only keep if brute-force possibilities are greater than this word's rank in the dictionary
    '''
    return [term for i, term in enumerate(terms) if 26**(len(term)) > i]