def emptyArray(key, add_label=None):
    """An array that starts empty"""

    result = {
        'key': key,
        'startEmpty': True
    }
    if add_label is not None:
        result['add'] = add_label
        result['style'] = {'add': 'btn-success'}
    return result