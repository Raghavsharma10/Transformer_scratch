def ini2value(ini_content):
    """
    INI FILE CONTENT TO Data
    """
    from mo_future import ConfigParser, StringIO

    buff = StringIO(ini_content)
    config = ConfigParser()
    config._read(buff, "dummy")

    output = {}
    for section in config.sections():
        output[section]=s = {}
        for k, v in config.items(section):
            s[k]=v
    return wrap(output)