def get_config(section, option, allow_empty_option=True, default=""):
    '''
    Get data from configs
    '''
    try:
        value = config.get(section, option)
        if value is None or len(value) == 0:
            if allow_empty_option:
                return ""
            else:
                return default
        else:
            return value
    except ConfigParser.NoSectionError:
        return default