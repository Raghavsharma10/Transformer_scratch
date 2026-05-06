def _asciify_list(data):
    """ Ascii-fies list values """
    ret = []
    for item in data:
        if isinstance(item, unicode):
            item = _remove_accents(item)
            item = item.encode('utf-8')
        elif isinstance(item, list):
            item = _asciify_list(item)
        elif isinstance(item, dict):
            item = _asciify_dict(item)
        ret.append(item)
    return ret