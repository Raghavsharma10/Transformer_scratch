def path_in_cache(self, filename, metahash):
        """Generates the path to a file in the mh cache.

        The generated path does not imply the file's existence!

        Args:
          filename: Filename relative to buildroot
          rule: A targets.SomeBuildRule object
          metahash: hash object
        """
        cpath = self._genpath(filename, metahash)
        if os.path.exists(cpath):
            return cpath
        else:
            raise CacheMiss