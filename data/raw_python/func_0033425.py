def rupdate(source, target):
    ''' recursively update nested dictionaries
        see: http://stackoverflow.com/a/3233356/1289080
    '''
    for k, v in target.iteritems():
        if isinstance(v, Mapping):
            r = rupdate(source.get(k, {}), v)
            source[k] = r
        else:
            source[k] = target[k]
    return source