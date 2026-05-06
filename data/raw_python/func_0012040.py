def keys(self, key=None, reverse=False):
        """sort the keys before returning them"""
        ks = sorted(list(dict.keys(self)), key=key, reverse=reverse)
        return ks