def wasSolvedBy(self, user):
        """Checks if this exercise has previously been solved by the user.

        """
        thisExercise = _Solution.what == self
        byThisUser = _Solution.who == user
        condition = q.AND(thisExercise, byThisUser)
        return self.store.query(_Solution, condition, limit=1).count() == 1