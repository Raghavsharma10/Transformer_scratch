def parse_querystring(querystring):
    """
    Return parsed querystring in dict
    """
    if querystring is None or len(querystring) == 0:
        return {}

    qs_dict = parse.parse_qs(querystring, keep_blank_values=True)
    for key in qs_dict:
        if len(qs_dict[key]) != 1:
            continue
        qs_dict[key] = qs_dict[key][0]
        if qs_dict[key] == '':
            qs_dict[key] = True

    return dict((key, qs_dict[key]) for key in qs_dict if len(key) != 0)