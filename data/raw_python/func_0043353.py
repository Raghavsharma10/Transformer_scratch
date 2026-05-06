def get_straat_by_id(self, id):
        '''
        Retrieve a `straat` by the Id.

        :param integer id: The id of the `straat`.
        :rtype: :class:`Straat`
        '''
        def creator():
            res = crab_gateway_request(
                self.client, 'GetStraatnaamWithStatusByStraatnaamId', id
            )
            if res == None:
                 raise GatewayResourceNotFoundException()
            return Straat(
                res.StraatnaamId,
                res.StraatnaamLabel,
                res.GemeenteId,
                res.StatusStraatnaam,
                res.Straatnaam,
                res.TaalCode,
                res.StraatnaamTweedeTaal,
                res.TaalCodeTweedeTaal,
                Metadata(
                    res.BeginDatum,
                    res.BeginTijd,
                    self.get_bewerking(res.BeginBewerking),
                    self.get_organisatie(res.BeginOrganisatie)
                )
            )

        if self.caches['long'].is_configured:
            key = 'GetStraatnaamWithStatusByStraatnaamId#%s' % (id)
            straat = self.caches['long'].get_or_create(key, creator)
        else:
            straat = creator()
        straat.set_gateway(self)
        return straat