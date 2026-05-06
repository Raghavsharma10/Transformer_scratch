def list_percelen_by_sectie(self, sectie):
        '''
        List all percelen in a `sectie`.

        :param sectie: The :class:`Sectie` for which the percelen are wanted.
        :param integer sort: Field to sort on.
        :rtype: A :class:`list` of :class:`Perceel`.
        '''
        sid = sectie.id
        aid = sectie.afdeling.id
        gid = sectie.afdeling.gemeente.id
        sectie.clear_gateway()

        def creator():
            url = self.base_url + '/municipality/%s/department/%s/section/%s/parcel' % (gid, aid, sid)
            h = self.base_headers
            p = {
                'data': 'adp'
            }
            res = capakey_rest_gateway_request(url, h, p).json()
            return [
                Perceel(
                    r['perceelnummer'],
                    sectie,
                    r['capakey'],
                    self.parse_percid(r['capakey']),
                ) for r in res['parcels']
            ]

        if self.caches['short'].is_configured:
            key = 'list_percelen_by_sectie_rest#%s#%s#%s' % (gid, aid, sid)
            percelen = self.caches['short'].get_or_create(key, creator)
        else:
            percelen = creator()
        for p in percelen:
            p.set_gateway(self)
        return percelen