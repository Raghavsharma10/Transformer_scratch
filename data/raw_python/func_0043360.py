def list_wegobjecten_by_straat(self, straat):
        '''
        List all `wegobjecten` in a :class:`Straat`

        :param straat: The :class:`Straat` for which the `wegobjecten` \
                are wanted.
        :rtype: A :class:`list` of :class:`Wegobject`
        '''
        try:
            id = straat.id
        except AttributeError:
            id = straat

        def creator():
            res = crab_gateway_request(
                self.client, 'ListWegobjectenByStraatnaamId', id
            )
            try:
                return [
                    Wegobject(
                        r.IdentificatorWegobject,
                        r.AardWegobject
                    )for r in res.WegobjectItem
                ]
            except AttributeError:
                return []
        if self.caches['short'].is_configured:
            key = 'ListWegobjectenByStraatnaamId#%s' % (id)
            wegobjecten = self.caches['short'].get_or_create(key, creator)
        else:
            wegobjecten = creator()
        for r in wegobjecten:
            r.set_gateway(self)
        return wegobjecten