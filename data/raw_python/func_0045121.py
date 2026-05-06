def get_user_ip(request):
    """Return user ip

    :param request: Django request object
    :return: user ip
    """
    ip = get_real_ip(request)
    if ip is None:
        ip = get_ip(request)
        if ip is None:
            ip = '127.0.0.1'
    return ip