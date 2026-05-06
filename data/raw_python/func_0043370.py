def get_subadres_by_id(self, id):
        '''
        Retrieve a `Subadres` by the Id.

        :param integer id: the Id of the `Subadres`
        :rtype: :class:`Subadres`
        '''
        def creator():
            res = crab_gateway_request(
                self.client, 'GetSubadresWithStatusBySubadresId', id
            )
            if res == None:
                raise GatewayResourceNotFoundException()
            return Subadres(
                res.SubadresId,
                res.Subadres,
                res.StatusSubadres,
                res.HuisnummerId,
                res.AardSubadres,
                Metadata(
                    res.BeginDatum,
                    res.BeginTijd,
                    self.get_bewerking(res.BeginBewerking),
                    self.get_organisatie(res.BeginOrganisatie)
                )
            )
        if self.caches['short'].is_configured:
            key = 'GetSubadresWithStatusBySubadresId#%s' % (id)
            subadres = self.caches['short'].get_or_create(key, creator)
        else:
            subadres = creator()
        subadres.set_gateway(self)
        return subadres