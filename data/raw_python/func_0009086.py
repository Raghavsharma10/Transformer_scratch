def make_levels_set(levels):
    '''make set efficient will convert all lists of items
    in levels to a set to speed up operations'''
    for level_key,level_filters in levels.items():
        levels[level_key] = make_level_set(level_filters)
    return levels