def solveAndNotify(self, request):
        """Notifies the owner of the current request (so, the user doing the
        exercise) that they've solved the exercise, and mark it as
        solved in the database.

        """
        remote = request.transport.remote
        withThisIdentifier = Exercise.identifier == self.exerciseIdentifier
        exercise = self.store.findUnique(Exercise, withThisIdentifier)
        solveAndNotify(remote, exercise)