def layer2dict(layer):
    """
    Return a json representation for a layer.
    """

    category = None
    username = None

    # bbox must be valid before proceeding
    if not layer.has_valid_bbox():
        message = 'Layer id: %s has a not valid bbox' % layer.id
        return None, message

    # we can proceed safely
    bbox = [float(layer.bbox_x0), float(layer.bbox_y0), float(layer.bbox_x1), float(layer.bbox_y1)]
    minX = bbox[0]
    minY = bbox[1]
    maxX = bbox[2]
    maxY = bbox[3]
    # coords hack needed by solr
    if (minX < -180):
        minX = -180
    if (maxX > 180):
        maxX = 180
    if (minY < -90):
        minY = -90
    if (maxY > 90):
        maxY = 90
    rectangle = box(minX, minY, maxX, maxY)
    wkt = "ENVELOPE({:f},{:f},{:f},{:f})".format(minX, maxX, maxY, minY)
    halfWidth = (maxX - minX) / 2.0
    halfHeight = (maxY - minY) / 2.0
    area = (halfWidth * 2) * (halfHeight * 2)
    domain = get_domain(layer.service.url)
    if hasattr(layer, 'layerwm'):
        category = layer.layerwm.category
        username = layer.layerwm.username
    abstract = layer.abstract
    if abstract:
        abstract = strip_tags(layer.abstract)
    else:
        abstract = ''
    if layer.type == "WM":
        originator = username
    else:
        originator = domain

    layer_dict = {
                    'id': layer.id,
                    'uuid': str(layer.uuid),
                    'type': 'Layer',
                    'layer_id': layer.id,
                    'name': layer.name,
                    'title': layer.title,
                    'layer_originator': originator,
                    'service_id': layer.service.id,
                    'service_type': layer.service.type,
                    'layer_category': category,
                    'layer_username': username,
                    'url': layer.url,
                    'keywords': [kw.name for kw in layer.keywords.all()],
                    'reliability': layer.reliability,
                    'recent_reliability': layer.recent_reliability,
                    'last_status': layer.last_status,
                    'is_public': layer.is_public,
                    'is_valid': layer.is_valid,
                    'availability': 'Online',
                    'location': '{"layerInfoPage": "' + layer.get_absolute_url + '"}',
                    'abstract': abstract,
                    'domain_name': layer.service.get_domain
                    }

    solr_date, date_type = get_date(layer)
    if solr_date is not None:
        layer_dict['layer_date'] = solr_date
        layer_dict['layer_datetype'] = date_type
    if bbox is not None:
        layer_dict['min_x'] = minX
        layer_dict['min_y'] = minY
        layer_dict['max_x'] = maxX
        layer_dict['max_y'] = maxY
        layer_dict['area'] = area
        layer_dict['bbox'] = wkt
        layer_dict['centroid_x'] = rectangle.centroid.x
        layer_dict['centroid_y'] = rectangle.centroid.y
        srs_list = [srs.encode('utf-8') for srs in layer.service.srs.values_list('code', flat=True)]
        layer_dict['srs'] = srs_list
    if layer.get_tile_url():
        layer_dict['tile_url'] = layer.get_tile_url()

    message = 'Layer %s successfully converted to json' % layer.id
    return layer_dict, message