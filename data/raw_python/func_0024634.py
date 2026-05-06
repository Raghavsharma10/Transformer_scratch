def data_import(self, json_response):
        """Import data from json response."""
        if 'data' not in json_response:
            raise PyVLXException('no element data found: {0}'.format(
                json.dumps(json_response)))
        data = json_response['data']

        for item in data:
            if 'category' not in item:
                raise PyVLXException('no element category: {0}'.format(
                    json.dumps(item)))
            category = item['category']
            if category == 'Window opener':
                self.load_window_opener(item)
            elif category in ['Roller shutter', 'Dual Shutter']:
                self.load_roller_shutter(item)
            elif category in ['Blind']:
                self.load_blind(item)
            else:
                self.pyvlx.logger.warning(
                    'WARNING: Could not parse product: %s', category)