def split_calls(func):
    """
    Decorator to split up server calls for methods using url parameters, due to the lenght
    limitation of the URI in Apache. By default 8190 bytes
    """
    def wrapper(*args, **kwargs):
        #The size limit is 8190 bytes minus url and api to call
        #For example (https://cmsweb-testbed.cern.ch:8443/dbs/prod/global/filechildren), so 192 bytes should be safe.
        size_limit = 8000
        encoded_url = urllib.urlencode(kwargs)
        if len(encoded_url) > size_limit:
            for key, value in kwargs.iteritems():
                ###only one (first) list at a time is splitted,
                ###currently only file lists are supported
                if key in ('logical_file_name', 'block_name', 'lumi_list', 'run_num') and isinstance(value, list):
                    ret_val = []
                    for splitted_param in list_parameter_splitting(data=dict(kwargs), #make a copy, since it is manipulated
                                                                   key=key,
                                                                   size_limit=size_limit):
                        try:
                            ret_val.extend(func(*args, **splitted_param))
                        except (TypeError, AttributeError):#update function call do not return lists
                            ret_val= []
                    return ret_val
            raise dbsClientException("Invalid input",
                                     "The lenght of the urlencoded parameters to API %s \
                                     is exceeding %s bytes and cannot be splitted." % (func.__name__, size_limit))
        else:
            return func(*args, **kwargs)
    return wrapper