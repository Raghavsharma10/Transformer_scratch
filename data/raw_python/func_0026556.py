def fieldset(title, items, options=None):
    """A field set with a title and sub items"""
    result = {
        'title': title,
        'type': 'fieldset',
        'items': items
    }
    if options is not None:
        result.update(options)

    return result