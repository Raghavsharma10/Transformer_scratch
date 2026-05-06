def create_layer_from_metadata_xml(resourcetype, xml, monitor=False, service=None, catalog=None):
    """
    Create a layer / keyword list from a metadata record if it does not already exist.
    """
    from models import gen_anytext, Layer

    if resourcetype == 'http://www.opengis.net/cat/csw/2.0.2':  # Dublin core
        md = CswRecord(etree.fromstring(xml))

    layer = Layer(
        is_monitored=monitor,
        name=md.title,
        title=md.title,
        abstract=md.abstract,
        xml=xml,
        service=service,
        catalog=catalog,
        anytext=gen_anytext(md.title, md.abstract, md.subjects)
    )

    if hasattr(md, 'alternative'):
        layer.name = md.alternative

    if md.bbox is not None:
        layer.bbox_x0 = format_float(md.bbox.minx)
        layer.bbox_y0 = format_float(md.bbox.miny)
        layer.bbox_x1 = format_float(md.bbox.maxx)
        layer.bbox_y1 = format_float(md.bbox.maxy)

        layer.wkt_geometry = bbox2wktpolygon([md.bbox.minx, md.bbox.miny, md.bbox.maxx, md.bbox.maxy])

    return layer, md.subjects