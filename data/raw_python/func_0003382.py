def config_keys(self, sortkey = False):
        """
        Return all configuration keys in this node, including configurations on children nodes.
        """
        if sortkey:
            items = sorted(self.items())
        else:
            items = self.items()
        for k,v in items:
            if isinstance(v, ConfigTree):
                for k2 in v.config_keys(sortkey):
                    yield k + '.' + k2
            else:
                yield k