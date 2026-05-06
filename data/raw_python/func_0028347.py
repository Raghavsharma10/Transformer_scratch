def version(request):
    """
    Returns the contents of version.json or a 404.
    """
    version_json = import_string(version_callback)(settings.BASE_DIR)
    if version_json is None:
        return HttpResponseNotFound('version.json not found')
    else:
        return JsonResponse(version_json)