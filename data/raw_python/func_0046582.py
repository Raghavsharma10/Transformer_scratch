def prepare_geojson(geojson):
    """
    Modifies incoming GeoJSON to make it Elastic friendly. This means:

        1. CW orientation of polygons.
        2. Re-casting of Features and FeatureCollections to Geometry and
           GeometryCollections.
    """
    # TODO CW orientation.
    geojson = deepcopy(geojson)

    if geojson["type"] == "Feature":
        geojson = geojson["geometry"]
        if hasattr(geojson, 'properties'):
            del geojson['properties']

    if geojson["type"] == "FeatureCollection":
        geojson["type"] = "GeometryCollection"
        geojson["geometries"] = [
            feature["geometry"] for feature in geojson["features"]
        ]
        del geojson["features"]

    return geojson