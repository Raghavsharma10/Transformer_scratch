def extract(query_dict, prefix=""):
    """
    Extract the *order_by*, *per_page*, and *page* parameters from
    `query_dict` (a Django QueryDict), and return a dict suitable for
    instantiating a preconfigured Table object.
    """

    strs = ['order_by']
    ints = ['per_page', 'page']

    extracted = { }

    for key in (strs + ints):
        if (prefix + key) in query_dict:
            val = query_dict.get(prefix + key)

            extracted[key] = (val
                if not key in ints
                else int(val))

    return extracted