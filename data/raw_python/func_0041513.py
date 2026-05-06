def read_config(filename=None):
    """
    Read a config filename into .ini format and return dict of shares.

    Keyword arguments:
    filename -- the path of config filename (default None)

    Return dict.
    """
    if not os.path.exists(filename):
        raise IOError('Impossibile trovare il filename %s' % filename)
    shares = []
    config = ConfigParser()
    config.read(filename)
    for share_items in [config.items(share_title) for share_title in
                        config.sections()]:
        dict_share = {}
        for key, value in share_items:
            if key == 'hostname' and '@' in value:
                hostname, credentials = (item[::-1] for item
                                         in value[::-1].split('@', 1))
                dict_share.update({key: hostname})
                credentials = tuple(cred.lstrip('"').rstrip('"')
                                    for cred in credentials.split(':', 1))
                dict_share.update({'username': credentials[0]})
                if len(credentials) > 1:
                    dict_share.update({'password': credentials[1]})
                continue
            dict_share.update({key: value})
        shares.append(dict_share)
    return shares