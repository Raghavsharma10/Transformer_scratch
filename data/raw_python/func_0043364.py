def get_terreinobject_by_id(self, id):
        '''
        Retrieve a `Terreinobject` by the Id.

        :param integer id: the Id of the `Terreinobject`
        :rtype: :class:`Terreinobject`
        '''
        def creator():
            res = crab_gateway_request(
                self.client,
                'GetTerreinobjectByIdentificatorTerreinobject', id
            )
            if res == None:
                raise GatewayResourceNotFoundException()
            return Terreinobject(
                res.IdentificatorTerreinobject,
                res.AardTerreinobject,
                (res.CenterX, res.CenterY),
                (res.MinimumX, res.MinimumY, res.MaximumX, res.MaximumY),
                Metadata(
                    res.BeginDatum,
                    res.BeginTijd,
                    self.get_bewerking(res.BeginBewerking),
                    self.get_organisatie(res.BeginOrganisatie)
                )
            )
        if self.caches['short'].is_configured:
            key = 'GetTerreinobjectByIdentificatorTerreinobject#%s' % (id)
            terreinobject = self.caches['short'].get_or_create(key, creator)
        else:
            terreinobject = creator()
        terreinobject.set_gateway(self)
        return terreinobject