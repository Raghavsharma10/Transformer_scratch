def list_secties_by_afdeling(self, afdeling):
        '''
        List all `secties` in a `kadastrale afdeling`.

        :param afdeling: The :class:`Afdeling` for which the `secties` are \
            wanted. Can also be the id of and `afdeling`.
        :rtype: A :class:`list` of `Sectie`.
        '''
        try:
            aid = afdeling.id
            gid = afdeling.gemeente.id
        except AttributeError:
            aid = afdeling
            afdeling = self.get_kadastrale_afdeling_by_id(aid)
            gid = afdeling.gemeente.id
        afdeling.clear_gateway()

        def creator():
            url = self.base_url + '/municipality/%s/department/%s/section' % (gid, aid)
            h = self.base_headers
            res = capakey_rest_gateway_request(url, h).json()
            return [
                Sectie(
                    r['sectionCode'],
                    afdeling
                ) for r in res['sections']
            ]

        if self.caches['long'].is_configured:
            key = 'list_secties_by_afdeling_rest#%s' % aid
            secties = self.caches['long'].get_or_create(key, creator)
        else:
            secties = creator()
        for s in secties:
            s.set_gateway(self)
        return secties