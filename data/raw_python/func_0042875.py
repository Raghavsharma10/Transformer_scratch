def resources_of_config(config):
    """ Returns all resources and models from config.
    """
    return set(             # unique values
        sum([               # join lists to flat list
            list(value)     # if value is iter (ex: list of resources)
            if hasattr(value, '__iter__')
            else [value, ]  # if value is not iter (ex: model or resource)
            for value in config.values()
        ], [])
    )