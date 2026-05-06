def get_perceel_by_capakey(self, capakey):
        '''
        Get a `perceel`.

        :param capakey: An capakey for a `perceel`.
        :rtype: :class:`Perceel`
        '''

        def creator():
            url = self.base_url + '/parcel/%s' % capakey
            h = self.base_headers
            p = {
                'geometry': 'full',
                'srs': '31370',
                'data': 'adp'
            }
            res = capakey_rest_gateway_request(url, h, p).json()
            return Perceel(
                res['perceelnummer'],
                Sectie(
                    res['sectionCode'],
                    Afdeling(
                        res['departmentCode'],
                        res['departmentName'],
                        Gemeente(res['municipalityCode'], res['municipalityName'])
                    )
                ),
                res['capakey'],
                Perceel.get_percid_from_capakey(res['capakey']),
                None,
                None,
                self._parse_centroid(res['geometry']['center']),
                self._parse_bounding_box(res['geometry']['boundingBox']),
                res['geometry']['shape']
            )

        if self.caches['short'].is_configured:
            key = 'get_perceel_by_capakey_rest#%s' % capakey
            perceel = self.caches['short'].get_or_create(key, creator)
        else:
            perceel = creator()
        perceel.set_gateway(self)
        return perceel