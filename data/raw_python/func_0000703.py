def contour_to_geojson(contour, geojson_filepath=None, min_angle_deg=None,
                       ndigits=5, unit='', stroke_width=1, geojson_properties=None, strdump=False,
                       serialize=True):
    """Transform matplotlib.contour to geojson."""
    collections = contour.collections
    contour_index = 0
    line_features = []
    for collection in collections:
        color = collection.get_edgecolor()
        for path in collection.get_paths():
            v = path.vertices
            if len(v) < 3:
                continue
            coordinates = keep_high_angle(v, min_angle_deg)
            if ndigits:
                coordinates = np.around(coordinates, ndigits)
            line = LineString(coordinates.tolist())
            properties = {
                "stroke-width": stroke_width,
                "stroke": rgb2hex(color[0]),
                "title": "%.2f" % contour.levels[contour_index] + ' ' + unit,
                "level-value": float("%.6f" % contour.levels[contour_index]),
                "level-index": contour_index
            }
            if geojson_properties:
                properties.update(geojson_properties)
            line_features.append(Feature(geometry=line, properties=properties))
        contour_index += 1
    feature_collection = FeatureCollection(line_features)
    return _render_feature_collection(feature_collection, geojson_filepath, strdump, serialize)