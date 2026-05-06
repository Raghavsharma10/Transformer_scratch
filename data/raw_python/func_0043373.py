def get_adrespositie_by_id(self, id):
        '''
        Retrieve a `Adrespositie` by the Id.

        :param integer id: the Id of the `Adrespositie`
        :rtype: :class:`Adrespositie`
        '''
        def creator():
            res = crab_gateway_request(
                self.client, 'GetAdrespositieByAdrespositieId', id
            )
            if res == None:
                raise GatewayResourceNotFoundException()
            return Adrespositie(
                res.AdrespositieId,
                res.HerkomstAdrespositie,
                res.Geometrie,
                res.AardAdres,
                Metadata(
                    res.BeginDatum,
                    res.BeginTijd,
                    self.get_bewerking(res.BeginBewerking),
                    self.get_organisatie(res.BeginOrganisatie)
                )
            )
        if self.caches['short'].is_configured:
            key = 'GetAdrespositieByAdrespositieId#%s' % (id)
            adrespositie = self.caches['short'].get_or_create(key, creator)
        else:
            adrespositie = creator()
        adrespositie.set_gateway(self)
        return adrespositie