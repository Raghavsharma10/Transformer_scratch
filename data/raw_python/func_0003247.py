def updatephysicalport(update_processor = partial(default_processor, excluding=('vhost', 'systemid',
                                                                                   'bridge', 'name'),
                                                                     disabled=('physicalnetwork',)),
                       reorder_dict = default_iterate_dict
                      ):
    """
    :param update_processor: update_processor(physcialport, walk, write, \*, parameters)    
    """
    def walker(walk, write, timestamp, parameters_dict):
        for key, parameters in reorder_dict(parameters_dict):
            try:
                value = walk(key)
            except KeyError:
                pass
            else:
                if update_processor(value, walk, write, parameters=parameters):
                    write(key, value)
    return walker