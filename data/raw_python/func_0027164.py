def get_ip_address(request):
    """
    Correct IP address is expected as first element of HTTP_X_FORWARDED_FOR or REMOTE_ADDR
    """
    if 'HTTP_X_FORWARDED_FOR' in request.META:
        return request.META['HTTP_X_FORWARDED_FOR'].split(',')[0].strip()
    else:
        return request.META['REMOTE_ADDR']