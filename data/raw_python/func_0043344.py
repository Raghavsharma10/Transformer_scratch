def list_provincies(self, gewest=2):
        '''
        List all `provincies` in a `gewest`.

        :param gewest: The :class:`Gewest` for which the \
            `provincies` are wanted.
        :param integer sort: What field to sort on.
        :rtype: A :class:`list` of :class:`Provincie`.
        '''
        try:
            gewest_id = gewest.id
        except AttributeError:
            gewest_id = gewest

        def creator():
            return [Provincie(p[0], p[1], Gewest(p[2])) for p in self.provincies if p[2] == gewest_id]

        if self.caches['permanent'].is_configured:
            key = 'ListProvinciesByGewestId#%s' % gewest_id
            provincies = self.caches['permanent'].get_or_create(key, creator)
        else:
            provincies = creator()
        for p in provincies:
            p.set_gateway(self)
        return provincies