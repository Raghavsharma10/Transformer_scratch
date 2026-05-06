def application_adapter(obj, request):
    """
    Adapter for rendering a :class:`pyramid_urireferencer.models.ApplicationResponse` to json.

    :param pyramid_urireferencer.models.ApplicationResponse obj: The response to be rendered.
    :rtype: :class:`dict`
    """
    return {
        'title': obj.title,
        'uri': obj.uri,
        'service_url': obj.service_url,
        'success': obj.success,
        'has_references': obj.has_references,
        'count': obj.count,
        'items': [{
                      'uri': i.uri,
                      'title': i.title
                  } for i in obj.items] if obj.items is not None else None
    }