def json_response(data, status=200):
    """Return a JsonResponse. Make sure you have django installed first."""
    from django.http import JsonResponse
    return JsonResponse(data=data, status=status, safe=isinstance(data, dict))