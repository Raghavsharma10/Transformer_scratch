def near(lat, lng, max_dist=None, unit_miles=False):
        """Find document near a point.

        For example:: find all document with in 25 miles radius from 32.0, -73.0.
        """
        filters = {
            "$nearSphere": {
                "$geometry": {
                    "type": "Point",
                    "coordinates": [lng, lat],
                }
            }
        }
        if max_dist:
            if unit_miles:  # pragma: no cover
                max_dist = max_dist / 1609.344
            filters["$nearSphere"]["$maxDistance"] = max_dist
        return filters