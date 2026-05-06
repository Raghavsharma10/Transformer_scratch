def list_terreinobjecten_by_huisnummer(self, huisnummer):
        '''
        List all `terreinobjecten` for a :class:`Huisnummer`

        :param huisnummer: The :class:`Huisnummer` for which the \
            `terreinobjecten` are wanted.
        :rtype: A :class:`list` of :class:`Terreinobject`
        '''
        try:
            id = huisnummer.id
        except AttributeError:
            id = huisnummer

        def creator():
            res = crab_gateway_request(
                self.client, 'ListTerreinobjectenByHuisnummerId', id
            )
            try:
                return[
                    Terreinobject(
                        r.IdentificatorTerreinobject,
                        r.AardTerreinobject
                    )for r in res.TerreinobjectItem
                ]
            except AttributeError:
                return []
        if self.caches['short'].is_configured:
            key = 'ListTerreinobjectenByHuisnummerId#%s' % (id)
            terreinobjecten = self.caches['short'].get_or_create(key, creator)
        else:
            terreinobjecten = creator()
        for r in terreinobjecten:
            r.set_gateway(self)
        return terreinobjecten