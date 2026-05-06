def make_level_set(level):
    '''make level set will convert one level into
    a set'''
    new_level = dict()
    for key,value in level.items():
        if isinstance(value,list):
            new_level[key] = set(value)
        else:
            new_level[key] = value
    return new_level