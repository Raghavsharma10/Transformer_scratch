def sethead(self, ref):
        """Set head to a git ref."""
        log.debug('[%s] Setting to ref %s', self.name, ref)
        try:
            ref = self.repo.rev_parse(ref)
        except gitdb.exc.BadObject:
            # Probably means we don't have it cached yet.
            # So maybe we can fetch it.
            ref = self.fetchref(ref)
        log.debug('[%s] Setting head to %s', self.name, ref)
        self.repo.head.reset(ref, working_tree=True)
        log.debug('[%s] Head object: %s', self.name, self.currenthead)