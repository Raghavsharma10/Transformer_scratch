def view(self, sort=None, purge=False, done=None, undone=None, **kwargs):
        """Handles the 'v' command.

        :sort: Sort pattern.
        :purge: Whether to purge items marked as 'done'.
        :done: Done pattern.
        :undone: Not done pattern.
        :kwargs: Additional arguments to pass to the View object.

        """
        View(self.model.modify(
            sort=self._getPattern(sort),
            purge=purge,
            done=self._getDone(done, undone)
        ), **kwargs)