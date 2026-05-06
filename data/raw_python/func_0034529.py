def index(request):
    """Handles a request based on method and calls the appropriate function"""
    if request.method == 'GET':
        return get(request)
    elif request.method == 'POST':
        return post(request)
    return HttpResponse('')