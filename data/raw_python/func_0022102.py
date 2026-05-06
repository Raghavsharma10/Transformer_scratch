def update_layers_esri_mapserver(service, greedy_opt=False):
    """
    Update layers for an ESRI REST MapServer.
    Sample endpoint: https://gis.ngdc.noaa.gov/arcgis/rest/services/SampleWorldCities/MapServer/?f=json
    """
    try:
        esri_service = ArcMapService(service.url)
        # set srs
        # both mapserver and imageserver exposes just one srs at the service level
        # not sure if other ones are supported, for now we just store this one

        # not sure why this is needed, for now commenting out
        # if wkt_text:
        #     params = {'exact': 'True', 'error': 'True', 'mode': 'wkt', 'terms': wkt_text}
        #     req = requests.get('http://prj2epsg.org/search.json', params=params)
        #     object = json.loads(req.content)
        #     srs = int(object['codes'][0]['code'])

        srs_code = esri_service.spatialReference.wkid
        srs, created = SpatialReferenceSystem.objects.get_or_create(code=srs_code)
        service.srs.add(srs)

        service.update_validity()

        # check if it has a WMS interface
        if 'supportedExtensions' in esri_service._json_struct and greedy_opt:
            if 'WMSServer' in esri_service._json_struct['supportedExtensions']:
                # we need to change the url
                # http://cga1.cga.harvard.edu/arcgis/rest/services/ecuador/ecuadordata/MapServer?f=pjson
                # http://cga1.cga.harvard.edu/arcgis/services/ecuador/
                # ecuadordata/MapServer/WMSServer?request=GetCapabilities&service=WMS
                wms_url = service.url.replace('/rest/services/', '/services/')
                if '?f=pjson' in wms_url:
                    wms_url = wms_url.replace('?f=pjson', 'WMSServer?')
                if '?f=json' in wms_url:
                    wms_url = wms_url.replace('?f=json', 'WMSServer?')
                LOGGER.debug('This ESRI REST endpoint has an WMS interface to process: %s' % wms_url)
                # import here as otherwise is circular (TODO refactor)
                from utils import create_service_from_endpoint
                create_service_from_endpoint(wms_url, 'OGC:WMS', catalog=service.catalog)
        # now process the REST interface
        layer_n = 0
        total = len(esri_service.layers)
        for esri_layer in esri_service.layers:
            # in some case the json is invalid
            # esri_layer._json_struct
            # {u'currentVersion': 10.01,
            # u'error':
            # {u'message': u'An unexpected error occurred processing the request.', u'code': 500, u'details': []}}
            if 'error' not in esri_layer._json_struct:
                LOGGER.debug('Updating layer %s' % esri_layer.name)
                layer, created = Layer.objects.get_or_create(
                    name=esri_layer.id,
                    service=service,
                    catalog=service.catalog
                )
                if layer.active:
                    layer.type = 'ESRI:ArcGIS:MapServer'
                    links = [[layer.type, service.url],
                             ['OGC:WMTS', settings.SITE_URL.rstrip('/') + '/' + layer.get_url_endpoint()]]
                    layer.title = esri_layer.name
                    layer.abstract = esri_service.serviceDescription
                    layer.url = service.url
                    layer.page_url = layer.get_absolute_url
                    links.append([
                        'WWW:LINK',
                        settings.SITE_URL.rstrip('/') + layer.page_url
                    ])
                    try:
                        layer.bbox_x0 = esri_layer.extent.xmin
                        layer.bbox_y0 = esri_layer.extent.ymin
                        layer.bbox_x1 = esri_layer.extent.xmax
                        layer.bbox_y1 = esri_layer.extent.ymax
                    except KeyError:
                        pass
                    try:
                        layer.bbox_x0 = esri_layer._json_struct['extent']['xmin']
                        layer.bbox_y0 = esri_layer._json_struct['extent']['ymin']
                        layer.bbox_x1 = esri_layer._json_struct['extent']['xmax']
                        layer.bbox_y1 = esri_layer._json_struct['extent']['ymax']
                    except Exception:
                        pass
                    layer.wkt_geometry = bbox2wktpolygon([layer.bbox_x0, layer.bbox_y0, layer.bbox_x1, layer.bbox_y1])
                    layer.xml = create_metadata_record(
                        identifier=str(layer.uuid),
                        source=service.url,
                        links=links,
                        format='ESRI:ArcGIS:MapServer',
                        type=layer.csw_type,
                        relation=service.id_string,
                        title=layer.title,
                        alternative=layer.title,
                        abstract=layer.abstract,
                        wkt_geometry=layer.wkt_geometry
                    )
                    layer.anytext = gen_anytext(layer.title, layer.abstract)
                    layer.save()
                    # dates
                    add_mined_dates(layer)
                layer_n = layer_n + 1
                # exits if DEBUG_SERVICES
                LOGGER.debug("Updating layer n. %s/%s" % (layer_n, total))
                if DEBUG_SERVICES and layer_n == DEBUG_LAYER_NUMBER:
                    return
    except Exception as err:
        message = "update_layers_esri_mapserver: {0}".format(
            err
        )
        check = Check(
            content_object=service,
            success=False,
            response_time=0,
            message=message
        )
        check.save()