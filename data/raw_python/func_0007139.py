def registry_adapter(obj, request):
    """
    Adapter for rendering a :class:`pyramid_urireferencer.models.RegistryResponse` to json.

    :param pyramid_urireferencer.models.RegistryResponse obj: The response to be rendered.
    :rtype: :class:`dict`
    """
    return {
        'query_uri': obj.query_uri,
        'success': obj.success,
        'has_references': obj.has_references,
        'count': obj.count,
        'applications': [{
                             'title': a.title,
                             'uri': a.uri,
                             'service_url': a.service_url,
                             'success': a.success,
                             'has_references': a.has_references,
                             'count': a.count,
                             'items': [{
                                           'uri': i.uri,
                                           'title': i.title
                                       } for i in a.items] if a.items is not None else None
                         } for a in obj.applications] if obj.applications is not None else None
    }