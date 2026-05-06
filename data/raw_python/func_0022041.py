def layer_mapproxy(request, catalog_slug, layer_uuid, path_info):
    """
    Get Layer with matching catalog and uuid
    """
    layer = get_object_or_404(Layer,
                              uuid=layer_uuid,
                              catalog__slug=catalog_slug)

    # for WorldMap layers we need to use the url of the layer
    if layer.service.type == 'Hypermap:WorldMap':
        layer.service.url = layer.url

    # Set up a mapproxy app for this particular layer
    mp, yaml_config = get_mapproxy(layer)

    query = request.META['QUERY_STRING']

    if len(query) > 0:
        path_info = path_info + '?' + query

    params = {}
    headers = {
            'X-Script-Name': '/registry/{0}/layer/{1}/map/'.format(catalog_slug, layer.id),
            'X-Forwarded-Host': request.META['HTTP_HOST'],
            'HTTP_HOST': request.META['HTTP_HOST'],
            'SERVER_NAME': request.META['SERVER_NAME'],
            }

    if path_info == '/config':
        response = HttpResponse(yaml_config, content_type='text/plain')
        return response

    # Get a response from MapProxy as if it was running standalone.
    mp_response = mp.get(path_info, params, headers)

    # Create a Django response from the MapProxy WSGI response.
    response = HttpResponse(mp_response.body, status=mp_response.status_int)
    for header, value in mp_response.headers.iteritems():
        response[header] = value

    return response