def get_gemeente_by_id(self, id):
        '''
        Retrieve a `gemeente` by the crab id.

        :param integer id: The CRAB id of the gemeente.
        :rtype: :class:`Gemeente`
        '''
        def creator():
            res = crab_gateway_request(
                self.client, 'GetGemeenteByGemeenteId', id
            )
            if res == None:
                 raise GatewayResourceNotFoundException()
            return Gemeente(
                res.GemeenteId,
                res.GemeenteNaam,
                res.NisGemeenteCode,
                Gewest(res.GewestId),
                res.TaalCode,
                (res.CenterX, res.CenterY),
                (res.MinimumX, res.MinimumY, res.MaximumX, res.MaximumY),
                Metadata(
                    res.BeginDatum,
                    res.BeginTijd,
                    self.get_bewerking(res.BeginBewerking),
                    self.get_organisatie(res.BeginOrganisatie)
                )
            )
        if self.caches['long'].is_configured:
            key = 'GetGemeenteByGemeenteId#%s' % id
            gemeente = self.caches['long'].get_or_create(key, creator)
        else:
            gemeente = creator()
        gemeente.set_gateway(self)
        return gemeente