def get_subnet_id(self, name):
        """
        Return subnet ID for given ``name``, if it exists.

        E.g. with a subnet mapping of ``{'abc123': 'ops', '67fd56': 'prod'}``,
        ``get_subnet_id('ops')`` would return ``'abc123'``. If the map has
        non-unique values, the first matching key will be returned.

        If no match is found, the given ``name`` is returned as-is. This works
        well for e.g. normalizing names-or-IDs to just IDs.
        """
        for subnet_id, subnet_name in self.config['subnets'].iteritems():
            if subnet_name == name:
                return subnet_id
        return name