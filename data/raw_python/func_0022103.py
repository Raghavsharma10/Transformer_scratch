def update_layers_esri_imageserver(service):
    """
    Update layers for an ESRI REST ImageServer.
    Sample endpoint: https://gis.ngdc.noaa.gov/arcgis/rest/services/bag_bathymetry/ImageServer/?f=json
    """
    try:
        esri_service = ArcImageService(service.url)
        # set srs
        # both mapserver and imageserver exposes just one srs at the service level
        # not sure if other ones are supported, for now we just store this one
        obj = json.loads(esri_service._contents)
        srs_code = obj['spatialReference']['wkid']
        srs, created = SpatialReferenceSystem.objects.get_or_create(code=srs_code)
        service.srs.add(srs)

        service.update_validity()

        layer, created = Layer.objects.get_or_create(name=obj['name'], service=service, catalog=service.catalog)
        if layer.active:
            layer.type = 'ESRI:ArcGIS:ImageServer'
            links = [[layer.type, service.url],
                     ['OGC:WMTS', settings.SITE_URL.rstrip('/') + '/' + layer.get_url_endpoint()]]
            layer.title = obj['name']
            layer.abstract = esri_service.serviceDescription
            layer.url = service.url
            layer.bbox_x0 = str(obj['extent']['xmin'])
            layer.bbox_y0 = str(obj['extent']['ymin'])
            layer.bbox_x1 = str(obj['extent']['xmax'])
            layer.bbox_y1 = str(obj['extent']['ymax'])
            layer.page_url = layer.get_absolute_url
            links.append([
                'WWW:LINK',
                settings.SITE_URL.rstrip('/') + layer.page_url
            ])
            layer.wkt_geometry = bbox2wktpolygon([layer.bbox_x0, layer.bbox_y0, layer.bbox_x1, layer.bbox_y1])
            layer.xml = create_metadata_record(
                identifier=str(layer.uuid),
                source=service.url,
                links=links,
                format='ESRI:ArcGIS:ImageServer',
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
    except Exception as err:
        message = "update_layers_esri_imageserver: {0}".format(
            err
        )
        check = Check(
            content_object=service,
            success=False,
            response_time=0,
            message=message
        )
        check.save()