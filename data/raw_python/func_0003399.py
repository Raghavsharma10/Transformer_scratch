def config_value_items(self, sortkey = False):
        """
        Return `(key, value)` tuples for all mapped configurations for this object
        """
        return ((k, getattr(self, k)) for k in self.config_value_keys(sortkey))