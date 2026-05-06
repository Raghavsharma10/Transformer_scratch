def getPutData(request):
    """Adds raw post to the PUT and DELETE querydicts on the request so they behave like post

    :param request: Request object to add PUT/DELETE to
    :type request: Request
    """
    dataDict = {}
    data = request.body

    for n in urlparse.parse_qsl(data):
        dataDict[n[0]] = n[1]

    setattr(request, 'PUT', dataDict)
    setattr(request, 'DELETE', dataDict)