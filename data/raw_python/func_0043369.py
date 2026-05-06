def list_subadressen_by_huisnummer(self, huisnummer):
        '''
        List all `subadressen` for a :class:`Huisnummer`.

        :param huisnummer: The :class:`Huisnummer` for which the \
            `subadressen` are wanted. OR A huisnummer id.
        :rtype: A :class:`list` of :class:`Gebouw`
        '''
        try:
            id = huisnummer.id
        except AttributeError:
            id = huisnummer

        def creator():
            res = crab_gateway_request(
                self.client, 'ListSubadressenWithStatusByHuisnummerId', id
            )
            try:
                return [ Subadres(
                    r.SubadresId,
                    r.Subadres,
                    r.StatusSubadres
                )for r in res.SubadresWithStatusItem ]
            except AttributeError:
                return []
        if self.caches['short'].is_configured:
            key = 'ListSubadressenWithStatusByHuisnummerId#%s' % (id)
            subadressen = self.caches['short'].get_or_create(key, creator)
        else:
            subadressen = creator()
        for s in subadressen:
            s.set_gateway(self)
        return subadressen