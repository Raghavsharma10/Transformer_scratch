def get_postadres_by_subadres(self, subadres):
        '''
        Get the `postadres` for a :class:`Subadres`.

        :param subadres: The :class:`Subadres` for which the \
            `postadres` is wanted. OR A subadres id.
        :rtype: A :class:`str`.
        '''
        try:
            id = subadres.id
        except AttributeError:
            id = subadres
        def creator():
            res = crab_gateway_request(
                self.client, 'GetPostadresBySubadresId', id
            )
            if res == None:
                 raise GatewayResourceNotFoundException()
            return res.Postadres
        if self.caches['short'].is_configured:
            key = 'GetPostadresBySubadresId#%s' % (id)
            postadres = self.caches['short'].get_or_create(key, creator)
        else:
            postadres = creator()
        return postadres