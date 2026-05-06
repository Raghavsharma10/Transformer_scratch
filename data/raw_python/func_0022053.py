def get_service(raw_xml):
    """
    Set a service object based on the XML metadata
       <dct:references scheme="OGC:WMS">http://ngamaps.geointapps.org/arcgis
       /services/RIO/Rio_Foundation_Transportation/MapServer/WMSServer
       </dct:references>
    :param instance:
    :return: Layer
    """
    from pycsw.core.etree import etree

    parsed = etree.fromstring(raw_xml, etree.XMLParser(resolve_entities=False))

    # <dc:format>OGC:WMS</dc:format>
    source_tag = parsed.find("{http://purl.org/dc/elements/1.1/}source")
    # <dc:source>
    #    http://ngamaps.geointapps.org/arcgis/services/RIO/Rio_Foundation_Transportation/MapServer/WMSServer
    # </dc:source>
    format_tag = parsed.find("{http://purl.org/dc/elements/1.1/}format")

    service_url = None
    service_type = None

    if hasattr(source_tag, 'text'):
        service_url = source_tag.text

    if hasattr(format_tag, 'text'):
        service_type = format_tag.text

    if hasattr(format_tag, 'text'):
        service_type = format_tag.text

    service, created = Service.objects.get_or_create(url=service_url,
                                                     is_monitored=False,
                                                     type=service_type)
    # TODO: dont hardcode SRS, get them from the parsed XML.
    srs, created = SpatialReferenceSystem.objects.get_or_create(code="EPSG:4326")
    service.srs.add(srs)

    return service