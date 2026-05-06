def parse_hstring(hs):
    """
    Parse a single item from the telescope server into name, value, comment.
    """
    # split the string on = and /, also stripping whitespace and annoying quotes
    name, value, comment = yield_three(
        [val.strip().strip("'") for val in filter(None, re.split("[=/]+", hs))]
    )

    # if comment has a slash in it, put it back together
    try:
        len(comment)
    except:
        pass
    else:
        comment = '/'.join(comment)
    return name, value, comment