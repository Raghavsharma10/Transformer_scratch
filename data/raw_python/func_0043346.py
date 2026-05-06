def list_gemeenten_by_provincie(self, provincie):
        '''
        List all `gemeenten` in a `provincie`.

        :param provincie: The :class:`Provincie` for which the \
            `gemeenten` are wanted.
        :rtype: A :class:`list` of :class:`Gemeente`.
        '''
        try:
            gewest = provincie.gewest
            prov = provincie
        except AttributeError:
            prov = self.get_provincie_by_id(provincie)
            gewest = prov.gewest
        gewest.clear_gateway()

        def creator():
            gewest_gemeenten = self.list_gemeenten(gewest.id)
            return[
                Gemeente(
                    r.id,
                    r.naam,
                    r.niscode,
                    gewest
                )for r in gewest_gemeenten if str(r.niscode)[0] == str(prov.niscode)[0]
            ]

        if self.caches['permanent'].is_configured:
            key = 'ListGemeentenByProvincieId#%s' % prov.id
            gemeente = self.caches['long'].get_or_create(key, creator)
        else:
            gemeente = creator()
        for g in gemeente:
            g.set_gateway(self)
        return gemeente