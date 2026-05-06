def remove_unicode_dict(input_dict):
    '''remove unicode keys and values from dict, encoding in utf8
    '''
    if isinstance(input_dict, collections.Mapping):
        return dict(map(remove_unicode_dict, input_dict.iteritems()))
    elif isinstance(input_dict, collections.Iterable):
        return type(input_dict)(map(remove_unicode_dict, input_dict))
    else:
        return input_dict