def get_wegsegment_by_id(self, id):
        '''
        Retrieve a `wegsegment` by the Id.

        :param integer id: the Id of the `wegsegment`
        :rtype: :class:`Wegsegment`
        '''
        def creator():
            res = crab_gateway_request(
                self.client,
                'GetWegsegmentByIdentificatorWegsegment', id
            )
            if res == None:
                raise GatewayResourceNotFoundException()
            return Wegsegment(
                res.IdentificatorWegsegment,
                res.StatusWegsegment,
                res.GeometriemethodeWegsegment,
                res.Geometrie,
                Metadata(
                    res.BeginDatum,
                    res.BeginTijd,
                    self.get_bewerking(res.BeginBewerking),
                    self.get_organisatie(res.BeginOrganisatie)
                )
            )
        if self.caches['short'].is_configured:
            key = 'GetWegsegmentByIdentificatorWegsegment#%s' % (id)
            wegsegment = self.caches['short'].get_or_create(key, creator)
        else:
            wegsegment = creator()
        wegsegment.set_gateway(self)
        return wegsegment