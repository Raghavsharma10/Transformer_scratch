def list_gewesten(self, sort=1):
        '''
        List all `gewesten` in Belgium.

        :param integer sort: What field to sort on.
        :rtype: A :class`list` of class: `Gewest`.
        '''
        def creator():
            res = crab_gateway_request(self.client, 'ListGewesten', sort)
            tmp = {}
            for r in res.GewestItem:
                if r.GewestId not in tmp:
                    tmp[r.GewestId] = {}
                tmp[r.GewestId][r.TaalCodeGewestNaam] = r.GewestNaam
            return[
                Gewest(
                    k,
                    v
                )for k, v in tmp.items()
            ]
        if self.caches['permanent'].is_configured:
            key = 'ListGewesten#%s' % sort
            gewesten = self.caches['permanent'].get_or_create(key, creator)
        else:
            gewesten = creator()
        for g in gewesten:
            g.set_gateway(self)
        return gewesten