def quoteattrs(data):
    '''Takes dict of attributes and returns their HTML representation'''
    items = []
    for key, value in data.items():
        items.append('{}={}'.format(key, quoteattr(value)))
    return ' '.join(items)