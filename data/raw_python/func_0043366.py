def get_perceel_by_id(self, id):
        '''
        Retrieve a `Perceel` by the Id.

        :param string id: the Id of the `Perceel`
        :rtype: :class:`Perceel`
        '''
        def creator():
            res = crab_gateway_request(
                self.client, 'GetPerceelByIdentificatorPerceel', id
            )
            if res == None:
                raise GatewayResourceNotFoundException()
            return Perceel(
                res.IdentificatorPerceel,
                (res.CenterX, res.CenterY),
                Metadata(
                    res.BeginDatum,
                    res.BeginTijd,
                    self.get_bewerking(res.BeginBewerking),
                    self.get_organisatie(res.BeginOrganisatie)
                )
            )
        if self.caches['short'].is_configured:
            key = 'GetPerceelByIdentificatorPerceel#%s' % (id)
            perceel = self.caches['short'].get_or_create(key, creator)
        else:
            perceel = creator()
        perceel.set_gateway(self)
        return perceel