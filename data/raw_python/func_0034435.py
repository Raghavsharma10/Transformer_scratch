def getClientIP(request):
    """Returns the best IP address found from the request"""
    forwardedfor = request.META.get('HTTP_X_FORWARDED_FOR')
    if forwardedfor:
        ip = forwardedfor.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')

    return ip