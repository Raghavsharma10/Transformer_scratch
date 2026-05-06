def get_wegobject_by_id(self, id):
        '''
        Retrieve a `Wegobject` by the Id.

        :param integer id: the Id of the `Wegobject`
        :rtype: :class:`Wegobject`
        '''
        def creator():
            res = crab_gateway_request(
                self.client, 'GetWegobjectByIdentificatorWegobject', id
            )
            if res == None:
                raise GatewayResourceNotFoundException()
            return Wegobject(
                res.IdentificatorWegobject,
                res.AardWegobject,
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
            key = 'GetWegobjectByIdentificatorWegobject#%s' % (id)
            wegobject = self.caches['short'].get_or_create(key, creator)
        else:
            wegobject = creator()
        wegobject.set_gateway(self)
        return wegobject