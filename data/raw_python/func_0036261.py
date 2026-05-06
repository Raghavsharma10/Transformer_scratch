def trim_filter(filter, levels=1):
    '''Trim @ref levels levels from the front of each path in @filter.'''
    trimmed = [f[levels:] for f in filter]
    return [f for f in trimmed if f]