def update_layers_wmts(service):
    """
    Update layers for an OGC:WMTS service.
    Sample endpoint: http://map1.vis.earthdata.nasa.gov/wmts-geo/1.0.0/WMTSCapabilities.xml
    """
    try:
        wmts = WebMapTileService(service.url)

        # set srs
        # WMTS is always in 4326
        srs, created = SpatialReferenceSystem.objects.get_or_create(code='EPSG:4326')
        service.srs.add(srs)

        service.update_validity()

        layer_names = list(wmts.contents)
        layer_n = 0
        total = len(layer_names)
        for layer_name in layer_names:
            ows_layer = wmts.contents[layer_name]
            LOGGER.debug('Updating layer %s' % ows_layer.name)
            layer, created = Layer.objects.get_or_create(name=ows_layer.name, service=service, catalog=service.catalog)
            if layer.active:
                links = [['OGC:WMTS', service.url],
                         ['OGC:WMTS', settings.SITE_URL.rstrip('/') + '/' + layer.get_url_endpoint()]]
                layer.type = 'OGC:WMTS'
                layer.title = ows_layer.title
                layer.abstract = ows_layer.abstract
                # keywords
                # @tomkralidis wmts does not seem to support this attribute
                keywords = None
                if hasattr(ows_layer, 'keywords'):
                    keywords = ows_layer.keywords
                    for keyword in keywords:
                        layer.keywords.add(keyword)
                layer.url = service.url
                layer.page_url = layer.get_absolute_url
                links.append([
                    'WWW:LINK',
                    settings.SITE_URL.rstrip('/') + layer.page_url
                ])
                bbox = list(ows_layer.boundingBoxWGS84 or (-179.0, -89.0, 179.0, 89.0))
                layer.bbox_x0 = bbox[0]
                layer.bbox_y0 = bbox[1]
                layer.bbox_x1 = bbox[2]
                layer.bbox_y1 = bbox[3]
                layer.wkt_geometry = bbox2wktpolygon(bbox)
                layer.xml = create_metadata_record(
                    identifier=str(layer.uuid),
                    source=service.url,
                    links=links,
                    format='OGC:WMS',
                    type=layer.csw_type,
                    relation=service.id_string,
                    title=ows_layer.title,
                    alternative=ows_layer.name,
                    abstract=layer.abstract,
                    keywords=keywords,
                    wkt_geometry=layer.wkt_geometry
                )
                layer.anytext = gen_anytext(layer.title, layer.abstract, keywords)
                layer.save()
                # dates
                add_mined_dates(layer)
            layer_n = layer_n + 1
            # exits if DEBUG_SERVICES
            LOGGER.debug("Updating layer n. %s/%s" % (layer_n, total))
            if DEBUG_SERVICES and layer_n == DEBUG_LAYER_NUMBER:
                return
    except Exception as err:
        message = "update_layers_wmts: {0}".format(
            err
        )
        check = Check(
            content_object=service,
            success=False,
            response_time=0,
            message=message
        )
        check.save()