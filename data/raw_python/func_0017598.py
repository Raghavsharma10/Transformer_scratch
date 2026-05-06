def thumbnail_options(request):
    """
    Returns the requested ThumbnailOption as JSON

    :param request: Request object
    :return: JSON serialized ThumbnailOption
    """
    response_data = [{'id': opt.pk, 'name': opt.name} for opt in ThumbnailOption.objects.all()]
    return http.HttpResponse(json.dumps(response_data), content_type="application/json")