def get(self, layer, where="1 = 1", fields=[], count_only=False, srid='4326'):
        """
        Gets a layer and returns it as honest to God GeoJSON.

        WHERE 1 = 1 causes us to get everything. We use OBJECTID in the WHERE clause
        to paginate, so don't use OBJECTID in your WHERE clause unless you're going to
        query under 1000 objects.
        """
        base_where = where
        # By default we grab all of the fields. Technically I think
        # we can just do "*" for all fields, but I found this was buggy in
        # the KMZ mode. I'd rather be explicit.
        fields = fields or self.enumerate_layer_fields(layer)

        jsobj = self.get_json(layer, where, fields, count_only, srid)

        # Sometimes you just want to know how far there is to go.
        if count_only:
            return jsobj.get('count')

        # If there is no geometry, we default to assuming it's a Table type
        # data format, and we dump a simple (non-geo) json of all of the data.
        if not jsobj.get('geometryType', None):
            return self.getTable(layer, where, fields, jsobj=jsobj)

        # From what I can tell, the entire layer tends to be of the same type,
        # so we only have to determine the parsing function once.
        geom_parser = self._determine_geom_parser(jsobj.get('geometryType'))

        features = []
        # We always want to run once, and then break out as soon as we stop
        # getting exceededTransferLimit.
        while True:
            features += [self.esri_to_geojson(feat, geom_parser) for feat in jsobj.get('features')]
            if jsobj.get('exceededTransferLimit', False) == False:
                break
            # If we've hit the transfer limit we offset by the last OBJECTID
            # returned and keep moving along.
            where = "%s > %s" % (self.object_id_field, features[-1]['properties'].get(self.object_id_field))
            if base_where != "1 = 1" :
                # If we have another WHERE filter we needed to tack that back on.
                where += " AND %s" % base_where
            jsobj = self.get_json(layer, where, fields, count_only, srid)

        return {
            'type': "FeatureCollection",
            'features': features
        }