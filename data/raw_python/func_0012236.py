def from_geojson(cls, data):
        """
        Return a Route from a GeoJSON dictionary, as returned by Route.geojson()

        """
        properties = data['properties']
        distance = properties.pop('distance')
        duration = properties.pop('duration')

        maneuvers = []
        for feature in data['features']:
            geom = feature['geometry']
            if geom['type'] == 'LineString':
                coords = geom['coordinates']
            else:
                maneuvers.append(Maneuver.from_geojson(feature))

        return Route(coords, distance, duration, maneuvers, **properties)