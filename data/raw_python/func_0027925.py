def _safe_get(self, revision, key):
        """
        Get an answer data (vote or rationale) by revision

        Args:
            revision (int): the revision number for student answer, could be
                0 (original) or 1 (revised)
            key (str); key for retrieve answer data, could be VOTE_KEY or
                RATIONALE_KEY

        Returns:
            the answer data or None if revision doesn't exists
        """
        if self.has_revision(revision):
            return self.raw_answers[revision].get(key)
        else:
            return None