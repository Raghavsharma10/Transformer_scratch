def list_straten(self, gemeente, sort=1):
        '''
        List all `straten` in a `Gemeente`.

        :param gemeente: The :class:`Gemeente` for which the \
            `straten` are wanted.
        :rtype: A :class:`list` of :class:`Straat`
        '''
        try:
            id = gemeente.id
        except AttributeError:
            id = gemeente

        def creator():
            res = crab_gateway_request(
                self.client, 'ListStraatnamenWithStatusByGemeenteId',
                id, sort
            )
            try:
                return[
                    Straat(
                        r.StraatnaamId,
                        r.StraatnaamLabel,
                        id,
                        r.StatusStraatnaam
                    )for r in res.StraatnaamWithStatusItem
                ]
            except AttributeError:
                return []
        if self.caches['long'].is_configured:
            key = 'ListStraatnamenWithStatusByGemeenteId#%s%s' % (id, sort)
            straten = self.caches['long'].get_or_create(key, creator)
        else:
            straten = creator()
        for s in straten:
            s.set_gateway(self)
        return straten