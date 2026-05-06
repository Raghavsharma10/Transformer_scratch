def merge(self, ref_name: str):
        """
        Merges two refs

        Args:
            ref_name: ref to merge in the current one
        """
        if self.is_dirty():
            LOGGER.error('repository is dirty; cannot merge: %s', ref_name)
            sys.exit(-1)
        LOGGER.info('merging ref: "%s" into branch: %s', ref_name, self.get_current_branch())
        self.repo.git.merge(ref_name)