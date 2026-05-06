def set_sessid(sessid):
    """
    Save this current sessid in ``$HOME/.profrc``
    """
    filename = path.join(path.expanduser('~'), '.profrc')
    config = configparser.ConfigParser()
    config.read(filename)
    config.set('DEFAULT', 'Session', sessid)
    with open(filename, 'w') as configfile:
        print("write a new sessid")
        config.write(configfile)