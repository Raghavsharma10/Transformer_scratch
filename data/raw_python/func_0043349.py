def list_deelgemeenten(self, gewest=2):
        '''
        List all `deelgemeenten` in a `gewest`.

        :param gewest: The :class:`Gewest` for which the \
            `deelgemeenten` are wanted. Currently only Flanders is supported.
        :rtype: A :class:`list` of :class:`Deelgemeente`.
        '''
        try:
            gewest_id = gewest.id
        except AttributeError:
            gewest_id = gewest

        if gewest_id != 2:
            raise ValueError('Currently only deelgemeenten in Flanders are known.')

        def creator():
            return [Deelgemeente(dg['id'], dg['naam'], dg['gemeente_niscode']) for dg in self.deelgemeenten.values()]

        if self.caches['permanent'].is_configured:
            key = 'ListDeelgemeentenByGewestId#%s' % gewest_id
            deelgemeenten = self.caches['permanent'].get_or_create(key, creator)
        else:
            deelgemeenten = creator()
        for dg in deelgemeenten:
            dg.set_gateway(self)
        return deelgemeenten