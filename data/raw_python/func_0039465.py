def is_cached(self):
        """Returns true if this rule is already cached."""
        # TODO: cache by target+hash, not per file.
        try:
            for item in self.rule.output_files:
                log.info(item)
                self.cachemgr.in_cache(item, self._metahash())
        except cache.CacheMiss:
            log.info('[%s]: Not cached.', self.address)
            return False
        else:
            log.info('[%s]: found in cache.', self.address)
            return True