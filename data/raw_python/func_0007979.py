def make_url(contents, domain=DEFAULT_DOMAIN, force_gist=False,
             size_for_gist=MAX_URL_LEN):
    """
    Returns the URL to open given the domain and contents.

    If the file contents are large, an anonymous gist will be created.

    Parameters
    ----------
    contents
        * string - assumed to be GeoJSON
        * an object that implements __geo_interface__
            A FeatureCollection will be constructed with one feature,
            the object.
        * a sequence of objects that each implement __geo_interface__
            A FeatureCollection will be constructed with the objects
            as the features
    domain - string, default http://geojson.io
    force_gist - force gist creation regardless of file size.

    For more information about __geo_interface__ see:
    https://gist.github.com/sgillies/2217756

    If the contents are large, then a gist will be created.

    """
    contents = make_geojson(contents)
    if len(contents) <= size_for_gist and not force_gist:
        url = data_url(contents, domain)
    else:
        gist = _make_gist(contents)
        url = gist_url(gist.id, domain)

    return url