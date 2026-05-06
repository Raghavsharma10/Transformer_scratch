def kw_str_parse(a_string):
    """convert a string in the form 'a=b, c=d, e=f' to a dict"""
    try:
        return dict((k, eval(v.rstrip(',')))
                    for k, v in kw_list_re.findall(a_string))
    except (AttributeError, TypeError):
        if isinstance(a_string, collections.Mapping):
            return a_string
        return {}