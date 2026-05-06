def list_huisnummers_by_straat(self, straat, sort=1):
        '''
        List all `huisnummers` in a `Straat`.

        :param straat: The :class:`Straat` for which the \
            `huisnummers` are wanted.
        :rtype: A :class: `list` of :class:`Huisnummer`
        '''
        try:
            id = straat.id
        except AttributeError:
            id = straat

        def creator():
            res = crab_gateway_request(
                self.client, 'ListHuisnummersWithStatusByStraatnaamId',
                id, sort
            )
            try:
                return[
                    Huisnummer(
                        r.HuisnummerId,
                        r.StatusHuisnummer,
                        r.Huisnummer,
                        id
                    ) for r in res.HuisnummerWithStatusItem
                ]
            except AttributeError:
                return []
        if self.caches['short'].is_configured:
            key = 'ListHuisnummersWithStatusByStraatnaamId#%s%s' % (id, sort)
            huisnummers = self.caches['short'].get_or_create(key, creator)
        else:
            huisnummers = creator()
        for h in huisnummers:
            h.set_gateway(self)
        return huisnummers