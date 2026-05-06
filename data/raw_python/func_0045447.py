def solvedBy(self, user):
        """Stores that this user has just solved this exercise.

        You probably want to notify the user when this happens. For
        that, see ``solveAndNotify``.

        """
        _Solution(store=self.store, who=user, what=self)