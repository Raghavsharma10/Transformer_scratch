def get_gemeente_by_id(self, id):
        '''
        Retrieve a `gemeente` by id (the NIScode).

        :rtype: :class:`Gemeente`
        '''

        def creator():
            url = self.base_url + '/municipality/%s' % id
            h = self.base_headers
            p = {
                'geometry': 'full',
                'srs': '31370'
            }
            res = capakey_rest_gateway_request(url, h, p).json()
            return Gemeente(
                res['municipalityCode'],
                res['municipalityName'],
                self._parse_centroid(res['geometry']['center']),
                self._parse_bounding_box(res['geometry']['boundingBox']),
                res['geometry']['shape']
            )

        if self.caches['long'].is_configured:
            key = 'get_gemeente_by_id_rest#%s' % id
            gemeente = self.caches['long'].get_or_create(key, creator)
        else:
            gemeente = creator()
        gemeente.set_gateway(self)
        return gemeente