def in_cache(self, objpath, metahash):
        """Returns true if object is cached.

        Args:
          objpath: Filename relative to buildroot.
          metahash: hash object
        """
        try:
            self.path_in_cache(objpath, metahash)
            return True
        except CacheMiss:
            return False