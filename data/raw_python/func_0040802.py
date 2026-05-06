def error_response(message, status=400, code=None):
    """"Return error message(in dict)."""
    from django.http import JsonResponse
    data = {'message': message}
    if code:
        data['code'] = code
    LOG.error("Error response, status code is : {} | {}".format(status, data))
    return JsonResponse(data=data, status=status)