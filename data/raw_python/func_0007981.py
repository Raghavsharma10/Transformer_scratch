def data_url(contents, domain=DEFAULT_DOMAIN):
    """
    Return the URL for embedding the GeoJSON data in the URL hash

    Parameters
    ----------
    contents - string of GeoJSON
    domain - string, default http://geojson.io

    """
    url = (domain + '#data=data:application/json,' +
           urllib.parse.quote(contents))
    return url