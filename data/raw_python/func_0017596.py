def url_reverse(request):
    """
    Reverse the requested URL (passed via GET / POST as `url_name` parameter)

    :param request: Request object
    :return: The reversed path
    """
    if request.method in ('GET', 'POST'): 
        data = getattr(request, request.method) 
        url_name = data.get('url_name') 
        try: 
            path = urls.reverse(url_name, args=data.getlist('args')) 
            (view_func, args, kwargs) = urls.resolve(path)
            return http.HttpResponse(path, content_type='text/plain')
        except urls.NoReverseMatch: 
            return http.HttpResponse('Error', content_type='text/plain')
    return http.HttpResponseNotAllowed(('GET', 'POST'))