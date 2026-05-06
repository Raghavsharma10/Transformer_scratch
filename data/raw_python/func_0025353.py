def clean_tuple(t0, clean_item_fn=None):
    """
        Return a json-clean tuple. Will log info message for failures.
    """
    clean_item_fn = clean_item_fn if clean_item_fn else clean_item
    l = list()
    for index, item in enumerate(t0):
        cleaned_item = clean_item_fn(item)
        l.append(cleaned_item)
    return tuple(l)