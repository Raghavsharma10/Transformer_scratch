def get_user_agent():
    """ Obtain the default user agent string sent to the server after
    a successful handshake.
    """
    from sys import platform, version_info
    template = "neobolt/{} Python/{}.{}.{}-{}-{} ({})"
    fields = (version,) + tuple(version_info) + (platform,)
    return template.format(*fields)