def saturate_kwargs(keys, **kwargs):
    """
    Saturate all combinations of kwargs

    Args:
        keys: keys in kwargs that want to use process
        **kwargs: kwargs for func
    """
    # Validate if keys are in kwargs and if they are iterable
    if isinstance(keys, str): keys = [keys]
    keys = [k for k in keys if k in kwargs and hasattr(kwargs.get(k, None), '__iter__')]
    if len(keys) == 0: return []

    # Saturate coordinates of kwargs
    kw_corr = list(product(*(range(len(kwargs[k])) for k in keys)))

    # Append all possible values
    kw_arr = []
    for corr in kw_corr: kw_arr.append(
        dict(zip(keys, [kwargs[keys[i]][corr[i]] for i in range(len(keys))]))
    )

    # All combinations of kwargs of inputs
    for k in keys: kwargs.pop(k, None)
    kw_arr = [{**k, **kwargs} for k in kw_arr]

    return kw_arr