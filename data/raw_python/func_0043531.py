def get_perceel_by_id_and_sectie(self, id, sectie):
        '''
        Get a `perceel`.

        :param id: An id for a `perceel`.
        :param sectie: The :class:`Sectie` that contains the perceel.
        :rtype: :class:`Perceel`
        '''
        sid = sectie.id
        aid = sectie.afdeling.id
        gid = sectie.afdeling.gemeente.id
        sectie.clear_gateway()

        def creator():
            url = self.base_url + '/municipality/%s/department/%s/section/%s/parcel/%s' % (
            gid, aid, sid, id)
            h = self.base_headers
            p = {
                'geometry': 'full',
                'srs': '31370',
                'data': 'adp'
            }
            res = capakey_rest_gateway_request(url, h, p).json()
            return Perceel(
                res['perceelnummer'],
                sectie,
                res['capakey'],
                Perceel.get_percid_from_capakey(res['capakey']),
                None,
                None,
                self._parse_centroid(res['geometry']['center']),
                self._parse_bounding_box(res['geometry']['boundingBox']),
                res['geometry']['shape']
            )

        if self.caches['short'].is_configured:
            key = 'get_perceel_by_id_and_sectie_rest#%s#%s#%s' % (id, sectie.id, sectie.afdeling.id)
            perceel = self.caches['short'].get_or_create(key, creator)
        else:
            perceel = creator()
        perceel.set_gateway(self)
        return perceel