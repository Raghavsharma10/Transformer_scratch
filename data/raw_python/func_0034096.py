def modify(self, sort=None, purge=False, done=None, undone=None):
        """Handles the 'm' command.

        :sort: Sort pattern.
        :purge: Whether to purge items marked as 'done'.
        :done: Done pattern.
        :undone: Not done pattern.

        """
        self.model.modifyInPlace(
            sort=self._getPattern(sort),
            purge=purge,
            done=self._getDone(done, undone)
        )