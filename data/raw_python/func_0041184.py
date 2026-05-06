def format_dict(dic, format_list, separator=',', default_value=str):
    """
    Format dict to string passing a list of keys as order
    Args:
        lista: List with elements to clean duplicates.
    """

    dic = collections.defaultdict(default_value, dic)

    str_format = separator.join(["{" + "{}".format(head) + "}" for head in format_list])

    return str_format.format(**dic)