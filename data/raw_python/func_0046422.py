def errors(request, *args, **kwargs):
    """
    A dummy view that will throw errors.

    It'll throw any HTTP error that is contained in the search query.
    """
    search_term = request.GET.get('q', None)
    if '400' in search_term:
        return HttpResponseBadRequest(MESSAGE_400)
    elif '403' in search_term:
        return HttpResponseForbidden(MESSAGE_403)
    elif '404' in search_term:
        return HttpResponseNotFound(MESSAGE_404)
    elif '405' in search_term:
        return HttpResponseNotAllowed(['PATCH'], MESSAGE_405)
    return HttpResponseServerError(MESSAGE_500)