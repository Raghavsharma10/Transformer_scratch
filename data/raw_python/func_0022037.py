def detect_metadata_url_scheme(url):
    """detect whether a url is a Service type that HHypermap supports"""

    scheme = None
    url_lower = url.lower()

    if any(x in url_lower for x in ['wms', 'service=wms']):
        scheme = 'OGC:WMS'
    if any(x in url_lower for x in ['wmts', 'service=wmts']):
        scheme = 'OGC:WMTS'
    elif all(x in url for x in ['/MapServer', 'f=json']):
        scheme = 'ESRI:ArcGIS:MapServer'
    elif all(x in url for x in ['/ImageServer', 'f=json']):
        scheme = 'ESRI:ArcGIS:ImageServer'

    return scheme