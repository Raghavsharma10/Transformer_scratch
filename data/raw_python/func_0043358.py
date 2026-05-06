def get_postkanton_by_huisnummer(self, huisnummer):
        '''
        Retrieve a `postkanton` by the Huisnummer.

        :param huisnummer: The :class:`Huisnummer` for which the `postkanton` \
                is wanted.
        :rtype: :class:`Postkanton`
        '''
        try:
            id = huisnummer.id
        except AttributeError:
            id = huisnummer

        def creator():
            res = crab_gateway_request(
                self.client, 'GetPostkantonByHuisnummerId', id
            )
            if res == None:
                 raise GatewayResourceNotFoundException()
            return Postkanton(
                res.PostkantonCode
            )
        if self.caches['short'].is_configured:
            key = 'GetPostkantonByHuisnummerId#%s' % (id)
            postkanton = self.caches['short'].get_or_create(key, creator)
        else:
            postkanton = creator()
        postkanton.set_gateway(self)
        return postkanton