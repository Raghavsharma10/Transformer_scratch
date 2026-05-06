def set_json(self, value: object):
        '''
        A shortcut for set_content() with JSON objects.
        '''
        self.set_content(modjson.dumps(value, cls=ExtendedJSONEncoder),
                         content_type='application/json')