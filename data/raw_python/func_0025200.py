def to_geojson(self, filename, proj, metadata=None):
        """
        Output the data in the STObject to a geoJSON file.

        Args:
            filename: Name of the file
            proj: PyProj object for converting the x and y coordinates back to latitude and longitue values.
            metadata: Metadata describing the object to be included in the top-level properties.
        """
        if metadata is None:
            metadata = {}
        json_obj = {"type": "FeatureCollection", "features": [], "properties": {}}
        json_obj['properties']['times'] = self.times.tolist()
        json_obj['properties']['dx'] = self.dx
        json_obj['properties']['step'] = self.step
        json_obj['properties']['u'] = self.u.tolist()
        json_obj['properties']['v'] = self.v.tolist()
        for k, v in metadata.items():
            json_obj['properties'][k] = v
        for t, time in enumerate(self.times):
            feature = {"type": "Feature",
                       "geometry": {"type": "Polygon"},
                       "properties": {}}
            boundary_coords = self.boundary_polygon(time)
            lonlat = np.vstack(proj(boundary_coords[0], boundary_coords[1], inverse=True))
            lonlat_list = lonlat.T.tolist()
            if len(lonlat_list) > 0:
                lonlat_list.append(lonlat_list[0])
            feature["geometry"]["coordinates"] = [lonlat_list]
            for attr in ["timesteps", "masks", "x", "y", "i", "j"]:
                feature["properties"][attr] = getattr(self, attr)[t].tolist()
            feature["properties"]["attributes"] = {}
            for attr_name, steps in self.attributes.items():
                feature["properties"]["attributes"][attr_name] = steps[t].tolist()
            json_obj['features'].append(feature)
        file_obj = open(filename, "w")
        json.dump(json_obj, file_obj, indent=1, sort_keys=True)
        file_obj.close()
        return