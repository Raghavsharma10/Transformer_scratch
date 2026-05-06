def data_import(self, json_response):
        """Import scenes from JSON response."""
        if 'data' not in json_response:
            raise PyVLXException('no element data found: {0}'.format(
                json.dumps(json_response)))
        data = json_response['data']
        for item in data:
            self.load_scene(item)