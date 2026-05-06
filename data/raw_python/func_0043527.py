def get_kadastrale_afdeling_by_id(self, aid):
        '''
        Retrieve a 'kadastrale afdeling' by id.

        :param aid: An id of a `kadastrale afdeling`.
        :rtype: A :class:`Afdeling`.
        '''

        def creator():
            url = self.base_url + '/department/%s' % (aid)
            h = self.base_headers
            p = {
                'geometry': 'full',
                'srs': '31370'
            }
            res = capakey_rest_gateway_request(url, h, p).json()
            return Afdeling(
                id=res['departmentCode'],
                naam=res['departmentName'],
                gemeente=Gemeente(res['municipalityCode'], res['municipalityName']),
                centroid=self._parse_centroid(res['geometry']['center']),
                bounding_box=self._parse_bounding_box(res['geometry']['boundingBox']),
                shape=res['geometry']['shape']
            )

        if self.caches['long'].is_configured:
            key = 'get_kadastrale_afdeling_by_id_rest#%s' % aid
            afdeling = self.caches['long'].get_or_create(key, creator)
        else:
            afdeling = creator()
        afdeling.set_gateway(self)
        return afdeling