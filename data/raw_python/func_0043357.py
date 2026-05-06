def list_postkantons_by_gemeente(self, gemeente):
        '''
        List all `postkantons` in a :class:`Gemeente`

        :param gemeente: The :class:`Gemeente` for which the \
            `potkantons` are wanted.
        :rtype: A :class:`list` of :class:`Postkanton`
        '''
        try:
            id = gemeente.id
        except AttributeError:
            id = gemeente

        def creator():
            res = crab_gateway_request(
                self.client, 'ListPostkantonsByGemeenteId', id
            )
            try:
                return[
                    Postkanton(
                        r.PostkantonCode
                    )for r in res.PostkantonItem
                ]
            except AttributeError:
                return []
        if self.caches['long'].is_configured:
            key = 'ListPostkantonsByGemeenteId#%s' % (id)
            postkantons = self.caches['long'].get_or_create(key, creator)
        else:
            postkantons = creator()
        for r in postkantons:
            r.set_gateway(self)
        return postkantons