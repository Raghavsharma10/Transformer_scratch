def save(self, sortkey = True):
        """
        Save configurations to a list of strings
        """
        return [k + '=' + repr(v) for k,v in self.config_items(sortkey)]