def create_metadata_record(**kwargs):
    """
    Create a csw:Record XML document from harvested metadata
    """

    if 'srs' in kwargs:
        srs = kwargs['srs']
    else:
        srs = '4326'

    modified = '%sZ' % datetime.datetime.utcnow().isoformat().split('.')[0]

    nsmap = Namespaces().get_namespaces(['csw', 'dc', 'dct', 'ows'])

    e = etree.Element(nspath_eval('csw:Record', nsmap), nsmap=nsmap)

    etree.SubElement(e, nspath_eval('dc:identifier', nsmap)).text = kwargs['identifier']
    etree.SubElement(e, nspath_eval('dc:title', nsmap)).text = kwargs['title']
    if 'alternative' in kwargs:
        etree.SubElement(e, nspath_eval('dct:alternative', nsmap)).text = kwargs['alternative']
    etree.SubElement(e, nspath_eval('dct:modified', nsmap)).text = modified
    etree.SubElement(e, nspath_eval('dct:abstract', nsmap)).text = kwargs['abstract']
    etree.SubElement(e, nspath_eval('dc:type', nsmap)).text = kwargs['type']
    etree.SubElement(e, nspath_eval('dc:format', nsmap)).text = kwargs['format']
    etree.SubElement(e, nspath_eval('dc:source', nsmap)).text = kwargs['source']

    if 'relation' in kwargs:
        etree.SubElement(e, nspath_eval('dc:relation', nsmap)).text = kwargs['relation']

    if 'keywords' in kwargs:
        if kwargs['keywords'] is not None:
            for keyword in kwargs['keywords']:
                etree.SubElement(e, nspath_eval('dc:subject', nsmap)).text = keyword

    for link in kwargs['links']:
        etree.SubElement(e, nspath_eval('dct:references', nsmap), scheme=link[0]).text = link[1]

    bbox2 = loads(kwargs['wkt_geometry']).bounds
    bbox = etree.SubElement(e, nspath_eval('ows:BoundingBox', nsmap),
                            crs='http://www.opengis.net/def/crs/EPSG/0/%s' % srs,
                            dimensions='2')

    etree.SubElement(bbox, nspath_eval('ows:LowerCorner', nsmap)).text = '%s %s' % (bbox2[1], bbox2[0])
    etree.SubElement(bbox, nspath_eval('ows:UpperCorner', nsmap)).text = '%s %s' % (bbox2[3], bbox2[2])

    return etree.tostring(e, pretty_print=True)