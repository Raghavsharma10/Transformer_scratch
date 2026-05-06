def get_gebouw_by_id(self, id):
        '''
        Retrieve a `Gebouw` by the Id.

        :param integer id: the Id of the `Gebouw`
        :rtype: :class:`Gebouw`
        '''
        def creator():
            res = crab_gateway_request(
                self.client, 'GetGebouwByIdentificatorGebouw', id
            )
            if res == None:
                raise GatewayResourceNotFoundException()
            return Gebouw(
                res.IdentificatorGebouw,
                res.AardGebouw,
                res.StatusGebouw,
                res.GeometriemethodeGebouw,
                res.Geometrie,
                Metadata(
                    res.BeginDatum,
                    res.BeginTijd,
                    self.get_bewerking(res.BeginBewerking),
                    self.get_organisatie(res.BeginOrganisatie)
                )
            )
        if self.caches['short'].is_configured:
            key = 'GetGebouwByIdentificatorGebouw#%s' % (id)
            gebouw = self.caches['short'].get_or_create(key, creator)
        else:
            gebouw = creator()
        gebouw.set_gateway(self)
        return gebouw