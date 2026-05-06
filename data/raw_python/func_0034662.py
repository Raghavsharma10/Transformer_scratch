def filterObjects(request, obj_id):
    """
    Filters Gallery for the requested ImageVideo objects.  Returns a Result object with
    serialized objects
    """
    if int(obj_id) == 0:
        obj = None
    else:
        obj = Gallery.objects.get(pk=obj_id)

    isanonymous = request.user.is_anonymous()

    if isanonymous and obj is None:
        LOGGER.warn('There was an anonymous access attempt from {} to {}'.format(getClientIP(request), obj))
        raise PermissionDenied()

    if isanonymous and obj and obj.security != Gallery.PUBLIC:
        LOGGER.warn('There was an anonymous access attempt from {} to {}'.format(getClientIP(request), obj))
        raise PermissionDenied()

    tags = json.loads(request.GET.get('filters', '[[]]'))
    more = json.loads(request.GET.get('more', 'false'))
    orderby = request.GET.get('orderby', request.user.frog_prefs.get().json()['orderby'])

    tags = [t for t in tags if t]

    return _filter(request, obj, tags=tags, more=more, orderby=orderby)