def make_geojson(contents):
    """
    Return a GeoJSON string from a variety of inputs.
    See the documentation for make_url for the possible contents
    input.

    Returns
    -------
    GeoJSON string

    """
    if isinstance(contents, six.string_types):
        return contents

    if hasattr(contents, '__geo_interface__'):
        features = [_geo_to_feature(contents)]
    else:
        try:
            feature_iter = iter(contents)
        except TypeError:
            raise ValueError('Unknown type for input')

        features = []
        for i, f in enumerate(feature_iter):
            if not hasattr(f, '__geo_interface__'):
                raise ValueError('Unknown type at index {0}'.format(i))
            features.append(_geo_to_feature(f))

    data = {'type': 'FeatureCollection', 'features': features}
    return json.dumps(data)