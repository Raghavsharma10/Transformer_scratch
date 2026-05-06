def list_deelgemeenten_by_gemeente(self, gemeente):
        '''
        List all `deelgemeenten` in a `gemeente`.

        :param gemeente: The :class:`Gemeente` for which the \
            `deelgemeenten` are wanted. Currently only Flanders is supported.
        :rtype: A :class:`list` of :class:`Deelgemeente`.
        '''
        try:
            niscode = gemeente.niscode
        except AttributeError:
            niscode = gemeente

        def creator():
            return [
                Deelgemeente(dg['id'], dg['naam'], dg['gemeente_niscode'])
                for dg in self.deelgemeenten.values() if dg['gemeente_niscode'] == niscode
            ]

        if self.caches['permanent'].is_configured:
            key = 'ListDeelgemeentenByGemeenteId#%s' % niscode
            deelgemeenten = self.caches['permanent'].get_or_create(key, creator)
        else:
            deelgemeenten = creator()
        for dg in deelgemeenten:
            dg.set_gateway(self)
        return deelgemeenten