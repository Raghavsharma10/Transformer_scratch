def index(request, obj_id):
    """Handles a request based on method and calls the appropriate function"""
    if request.method == 'GET':
        return get(request, obj_id)
    elif request.method == 'PUT':
        getPutData(request)
        return put(request, obj_id)