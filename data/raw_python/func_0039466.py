def get_from_cache(self):
        """See if this rule has already been built and cached."""
        for item in self.rule.output_files:
            dstpath = os.path.join(self.buildroot, item)
            self.linkorcopy(
                self.cachemgr.path_in_cache(item, self._metahash()),
                dstpath)