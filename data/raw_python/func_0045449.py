def getExerciseDetails(self, identifier):
        """Gets the details for a particular exercise.

        """
        exercise = self._getExercise(identifier)
        response = {
            b"identifier": exercise.identifier,
            b"title": exercise.title,
            b"description": exercise.description,
            b"solved": exercise.wasSolvedBy(self.user)
        }
        return response