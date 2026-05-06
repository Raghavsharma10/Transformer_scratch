def groupby_dict(dictionary, key):
    ''' Group dict of dicts by key.
    '''
    return dict((k, list(g)) for k, g in itertools.groupby(sorted(dictionary.keys(), key=lambda name: dictionary[name][key]), key=lambda name: dictionary[name][key]))