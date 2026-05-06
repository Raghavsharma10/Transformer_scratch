def update_layers_warper(service):
    """
    Update layers for a Warper service.
    Sample endpoint: http://warp.worldmap.harvard.edu/maps
    """
    params = {'field': 'title', 'query': '', 'show_warped': '1', 'format': 'json'}
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    request = requests.get(service.url, headers=headers, params=params)

    try:
        records = json.loads(request.content)
        total_pages = int(records['total_pages'])

        # set srs
        # Warper supports only 4326, 900913, 3857
        for crs_code in ['EPSG:4326', 'EPSG:900913', 'EPSG:3857']:
            srs, created = SpatialReferenceSystem.objects.get_or_create(code=crs_code)
            service.srs.add(srs)

        service.update_validity()

        for i in range(1, total_pages + 1):
            params = {'field': 'title', 'query': '', 'show_warped': '1', 'format': 'json', 'page': i}
            request = requests.get(service.url, headers=headers, params=params)
            records = json.loads(request.content)
            LOGGER.debug('Fetched %s' % request.url)
            layers = records['items']
            layer_n = 0
            total = len(layers)
            for layer in layers:
                name = layer['id']
                title = layer['title']
                abstract = layer['description']
                bbox = layer['bbox']
                # dates
                dates = []
                if 'published_date' in layer:
                    dates.append(layer['published_date'])
                if 'date_depicted' in layer:
                    dates.append(layer['date_depicted'])
                if 'depicts_year' in layer:
                    dates.append(layer['depicts_year'])
                if 'issue_year' in layer:
                    dates.append(layer['issue_year'])
                layer, created = Layer.objects.get_or_create(name=name, service=service, catalog=service.catalog)
                if layer.active:
                    # update fields
                    # links = [['OGC:WMTS', settings.SITE_URL.rstrip('/') + '/' + layer.get_url_endpoint()]]
                    layer.type = 'Hypermap:WARPER'
                    layer.title = title
                    layer.abstract = abstract
                    layer.is_public = True
                    layer.url = '%s/wms/%s?' % (service.url, name)
                    layer.page_url = '%s/%s' % (service.url, name)
                    # bbox
                    x0 = None
                    y0 = None
                    x1 = None
                    y1 = None
                    if bbox:
                        bbox_list = bbox.split(',')
                        x0 = format_float(bbox_list[0])
                        y0 = format_float(bbox_list[1])
                        x1 = format_float(bbox_list[2])
                        y1 = format_float(bbox_list[3])
                    layer.bbox_x0 = x0
                    layer.bbox_y0 = y0
                    layer.bbox_x1 = x1
                    layer.bbox_y1 = y1
                    layer.save()
                    # dates
                    add_mined_dates(layer)
                    add_metadata_dates_to_layer(dates, layer)
                layer_n = layer_n + 1
                # exits if DEBUG_SERVICES
                LOGGER.debug("Updating layer n. %s/%s" % (layer_n, total))
                if DEBUG_SERVICES and layer_n == DEBUG_LAYER_NUMBER:
                    return

    except Exception as err:
        message = "update_layers_warper: {0}. request={1} response={2}".format(
            err,
            service.url,
            request.text
        )
        check = Check(
            content_object=service,
            success=False,
            response_time=0,
            message=message
        )
        check.save()