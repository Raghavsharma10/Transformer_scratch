def get_meta(meta, name):
    """Retrieves the metadata variable 'name' from the 'meta' dict."""
    assert name in meta
    data = meta[name]

    if data['t'] in ['MetaString', 'MetaBool']:
        return data['c']
    elif data['t'] == 'MetaInlines':
        # Handle bug in pandoc 2.2.3 and 2.2.3.1: Return boolean value rather
        # than strings, as appropriate.
        if len(data['c']) == 1 and data['c'][0]['t'] == 'Str':
            if data['c'][0]['c'] in ['true', 'True', 'TRUE']:
                return True
            elif data['c'][0]['c'] in ['false', 'False', 'FALSE']:
                return False
        return stringify(data['c'])
    elif data['t'] == 'MetaList':
        return [stringify(v['c']) for v in data['c']]
    else:
        raise RuntimeError("Could not understand metadata variable '%s'." %
                           name)