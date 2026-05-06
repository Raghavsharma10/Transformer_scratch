def dump_dict(dict_input, indent=4):
    """
    辞書型変数を文字列に変換して返す
    """

    dict_work = dict(dict_input)
    """
    for key, value in six.iteritems(dict_input):
        if any([f(value) for f in (is_float, isDict, is_list_or_tuple)]):
            dict_work[key] = value
            continue

        try:
            dict_work[key] = str(value)
        except:
            dict_work[key] = str(type(value))

        dict_work[key] = _convert_dump_dict(value)
    """

    try:
        import json
        return json.dumps(dict_work, sort_keys=True, indent=indent)
    except ImportError:
        pass

    try:
        import simplejson as json
        return json.dumps(dict_work, sort_keys=True, indent=indent)
    except ImportError:
        pass

    try:
        import pprint
        return pprint.pformat(dict_work, indent=indent)
    except ImportError:
        pass

    return str(dict_work)