def init_xena(api, logger, owner, ip=None, port=57911):
    """ Create XenaManager object.

    :param api: cli/rest
    :param logger: python logger
    :param owner: owner of the scripting session
    :param ip: rest server IP
    :param port: rest server TCP port
    :return: Xena object
    :rtype: XenaApp
    """

    if api == ApiType.socket:
        api_wrapper = XenaCliWrapper(logger)
    elif api == ApiType.rest:
        api_wrapper = XenaRestWrapper(logger, ip, port)
    return XenaApp(logger, owner, api_wrapper)