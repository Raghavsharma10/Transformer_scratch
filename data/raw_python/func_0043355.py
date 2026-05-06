def list_huisnummers_by_perceel(self, perceel, sort=1):
        '''
        List all `huisnummers` on a `Pereel`.

        Generally there will only be one, but multiples are possible.

        :param perceel: The :class:`Perceel` for which the \
            `huisnummers` are wanted.
        :rtype: A :class: `list` of :class:`Huisnummer`
        '''
        try:
            id = perceel.id
        except AttributeError:
            id = perceel

        def creator():
            res = crab_gateway_request(
                self.client, 'ListHuisnummersWithStatusByIdentificatorPerceel',
                id, sort
            )
            try:
                huisnummers= []
                for r in res.HuisnummerWithStatusItem:
                    h = self.get_huisnummer_by_id(r.HuisnummerId)
                    h.clear_gateway()
                    huisnummers.append(h)
                return huisnummers
            except AttributeError:
                return []
        if self.caches['short'].is_configured:
            key = 'ListHuisnummersWithStatusByIdentificatorPerceel#%s%s' % (id, sort)
            huisnummers = self.caches['short'].get_or_create(key, creator)
        else:
            huisnummers = creator()
        for h in huisnummers:
            h.set_gateway(self)
        return huisnummers