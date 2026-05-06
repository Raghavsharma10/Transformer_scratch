def list_percelen_by_huisnummer(self, huisnummer):
        '''
        List all `percelen` for a :class:`Huisnummer`

        :param huisnummer: The :class:`Huisnummer` for which the \
            `percelen` are wanted.
        :rtype: A :class:`list` of :class:`Perceel`
        '''
        try:
            id = huisnummer.id
        except AttributeError:
            id = huisnummer

        def creator():
            res = crab_gateway_request(
                self.client, 'ListPercelenByHuisnummerId', id
            )
            try:
                return [
                    Perceel(
                        r.IdentificatorPerceel
                    )for r in res.PerceelItem
                ]
            except AttributeError:
                return []
        if self.caches['short'].is_configured:
            key = 'ListPercelenByHuisnummerId#%s' % (id)
            percelen = self.caches['short'].get_or_create(key, creator)
        else:
            percelen = creator()
        for r in percelen:
            r.set_gateway(self)
        return percelen