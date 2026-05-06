def list_gebouwen_by_huisnummer(self, huisnummer):
        '''
        List all `gebouwen` for a :class:`Huisnummer`.

        :param huisnummer: The :class:`Huisnummer` for which the \
            `gebouwen` are wanted.
        :rtype: A :class:`list` of :class:`Gebouw`
        '''
        try:
            id = huisnummer.id
        except AttributeError:
            id = huisnummer

        def creator():
            res = crab_gateway_request(
                self.client, 'ListGebouwenByHuisnummerId', id
            )
            try:
                return [
                    Gebouw(
                        r.IdentificatorGebouw,
                        r.AardGebouw,
                        r.StatusGebouw
                    )for r in res.GebouwItem
                ]
            except AttributeError:
                return []
        if self.caches['short'].is_configured:
            key = 'ListGebouwenByHuisnummerId#%s' % (id)
            gebouwen = self.caches['short'].get_or_create(key, creator)
        else:
            gebouwen = creator()
        for r in gebouwen:
            r.set_gateway(self)
        return gebouwen