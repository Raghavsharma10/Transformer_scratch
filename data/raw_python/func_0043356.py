def get_huisnummer_by_id(self, id):
        '''
        Retrieve a `huisnummer` by the Id.

        :param integer id: the Id of the `huisnummer`
        :rtype: :class:`Huisnummer`
        '''
        def creator():
            res = crab_gateway_request(
                self.client, 'GetHuisnummerWithStatusByHuisnummerId', id
            )
            if res == None:
                 raise GatewayResourceNotFoundException()
            return Huisnummer(
                res.HuisnummerId,
                res.StatusHuisnummer,
                res.Huisnummer,
                res.StraatnaamId,
                Metadata(
                    res.BeginDatum,
                    res.BeginTijd,
                    self.get_bewerking(res.BeginBewerking),
                    self.get_organisatie(res.BeginOrganisatie)
                )
            )
        if self.caches['short'].is_configured:
            key = 'GetHuisnummerWithStatusByHuisnummerId#%s' % (id)
            huisnummer = self.caches['short'].get_or_create(key, creator)
        else:
            huisnummer = creator()
        huisnummer.set_gateway(self)
        return huisnummer