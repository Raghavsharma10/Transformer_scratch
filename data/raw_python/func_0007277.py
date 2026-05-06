def url_parse(name):
    """parse urls with different prefixes"""
    position = name.find("github.com")
    if position >= 0:
        if position != 0:
            position_1 = name.find("www.github.com")
            position_2 = name.find("http://github.com")
            position_3 = name.find("https://github.com")
            if position_1*position_2*position_3 != 0:
                exception()
                sys.exit(0)
        name = name[position+11:]
        if name.endswith('/'):
            name = name[:-1]
        return name
    else:
        if name.endswith('/'):
            name = name[:-1]
        return name