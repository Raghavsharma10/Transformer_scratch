def image(request, obj_id):
    """Handles a request based on method and calls the appropriate function"""
    obj = Image.objects.get(pk=obj_id)
    if request.method == 'POST':
        return post(request, obj)
    elif request.method == 'PUT':
        getPutData(request)
        return put(request, obj)
    elif request.method == 'DELETE':
        getPutData(request)
        return delete(request, obj)