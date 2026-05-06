def config_value_keys(self, sortkey = False):
        """
        Return configuration keys directly stored in this node. Configurations in child nodes are not included.
        """
        if sortkey:
            items = sorted(self.items())
        else:
            items = self.items()
        return (k for k,v in items if not isinstance(v,ConfigTree))