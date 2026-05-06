def list_gemeenten(self, sort=1):
        '''
        List all `gemeenten` in Vlaanderen.

        :param integer sort: What field to sort on.
        :rtype: A :class:`list` of :class:`Gemeente`.
        '''

        def creator():
            url = self.base_url + '/municipality'
            h = self.base_headers
            p = {
                'orderbyCode': sort == 1
            }
            res = capakey_rest_gateway_request(url, h, p).json()
            return [
                Gemeente(r['municipalityCode'], r['municipalityName'])
                for r in res['municipalities']
            ]

        if self.caches['permanent'].is_configured:
            key = 'list_gemeenten_rest#%s' % sort
            gemeente = self.caches['permanent'].get_or_create(key, creator)
        else:
            gemeente = creator()
        for g in gemeente:
            g.set_gateway(self)
        return gemeente