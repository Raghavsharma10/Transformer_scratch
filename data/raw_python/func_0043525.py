def list_kadastrale_afdelingen(self):
        '''
        List all `kadastrale afdelingen` in Flanders.

        :param integer sort: Field to sort on.
        :rtype: A :class:`list` of :class:`Afdeling`.
        '''

        def creator():
            gemeentes = self.list_gemeenten()
            res = []
            for g in gemeentes:
                res += self.list_kadastrale_afdelingen_by_gemeente(g)
            return res

        if self.caches['permanent'].is_configured:
            key = 'list_afdelingen_rest'
            afdelingen = self.caches['permanent'].get_or_create(key, creator)
        else:
            afdelingen = creator()
        return afdelingen