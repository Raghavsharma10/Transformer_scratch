def lookup_name(self, name):
        """Returns a struct if one exists"""

        try:
            info = self._get_by_name(self._source, name)
        except NotImplementedError:
            pass
        else:
            if info:
                return info
            return

        info = self.lookup_name_fast(name)
        if info:
            return info
        return self.lookup_name_slow(name)