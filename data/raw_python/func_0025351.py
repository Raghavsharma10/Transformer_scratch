def clean_dict(d0, clean_item_fn=None):
    """
        Return a json-clean dict. Will log info message for failures.
    """
    clean_item_fn = clean_item_fn if clean_item_fn else clean_item
    d = dict()
    for key in d0:
        cleaned_item = clean_item_fn(d0[key])
        if cleaned_item is not None:
            d[key] = cleaned_item
    return d