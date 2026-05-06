def contourf_to_geojson_overlap(contourf, geojson_filepath=None, min_angle_deg=None,
                                ndigits=5, unit='', stroke_width=1, fill_opacity=.9,
                                geojson_properties=None, strdump=False, serialize=True):
    """Transform matplotlib.contourf to geojson with overlapping filled contours."""
    polygon_features = []
    contourf_idx = 0
    for collection in contourf.collections:
        color = collection.get_facecolor()
        for path in collection.get_paths():
            for coord in path.to_polygons():
                if min_angle_deg:
                    coord = keep_high_angle(coord, min_angle_deg)
                coord = np.around(coord, ndigits) if ndigits else coord
                polygon = Polygon(coordinates=[coord.tolist()])
                fcolor = rgb2hex(color[0])
                properties = set_contourf_properties(stroke_width, fcolor, fill_opacity, contourf.levels, contourf_idx, unit)
                if geojson_properties:
                    properties.update(geojson_properties)
                feature = Feature(geometry=polygon, properties=properties)
                polygon_features.append(feature)
        contourf_idx += 1
    feature_collection = FeatureCollection(polygon_features)
    return _render_feature_collection(feature_collection, geojson_filepath, strdump, serialize)