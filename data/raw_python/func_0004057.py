def get_json(self, layer, where="1 = 1", fields=[], count_only=False, srid='4326'):
        """
        Gets the JSON file from ArcGIS
        """
        params = {
                'where': where,
                'outFields': ", ".join(fields),
                'returnGeometry': True,
                'outSR': srid,
                'f': "pjson",
                'orderByFields': self.object_id_field,
                'returnCountOnly': count_only
            }
        if self.token:
            params['token'] = self.token
        if self.geom_type:
            params.update({'geometryType': self.geom_type})
        response = requests.get(self._build_query_request(layer), params=params)
        return response.json()