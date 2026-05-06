def getboolean_config(section, option, default=False):
    '''
    Get data from configs which store boolean records
    '''
    try:
        return config.getboolean(section, option) or default
    except ConfigParser.NoSectionError:
        return default