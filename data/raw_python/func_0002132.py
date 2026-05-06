def _find_base_tds_url(catalog_url):
    """Identify the base URL of the THREDDS server from the catalog URL.

    Will retain URL scheme, host, port and username/password when present.
    """
    url_components = urlparse(catalog_url)
    if url_components.path:
        return catalog_url.split(url_components.path)[0]
    else:
        return catalog_url