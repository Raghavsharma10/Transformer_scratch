def index(request, obj_id=None):
    """Handles a request based on method and calls the appropriate function"""
    if request.method == 'GET':
        return get(request, obj_id)
    elif request.method == 'POST':
        return post(request)
    elif request.method == 'PUT':
        getPutData(request)
        return put(request, obj_id)
    elif request.method == 'DELETE':
        getPutData(request)
        return delete(request, obj_id)