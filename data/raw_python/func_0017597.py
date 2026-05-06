def url_image(request, image_id, thumb_options=None, width=None, height=None):
    """
    Converts a filer image ID in a complete path

    :param request: Request object
    :param image_id: Filer image ID
    :param thumb_options: ThumbnailOption ID
    :param width: user-provided width
    :param height: user-provided height
    :return: JSON serialized URL components ('url', 'width', 'height')
    """
    image = File.objects.get(pk=image_id)
    if getattr(image, 'canonical_url', None):
        url = image.canonical_url
    else:
        url = image.url
    thumb = _return_thumbnail(image, thumb_options, width, height)
    if thumb:
        image = thumb
        url = image.url
    data = {
        'url': url,
        'width': image.width,
        'height': image.height,
    }
    return http.HttpResponse(json.dumps(data), content_type='application/json')