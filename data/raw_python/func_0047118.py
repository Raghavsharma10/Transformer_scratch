def _autoset_min_reps_consistency(self):
        """
        Sets the program mode to 'weekly', 'daily' or 'exercise'
        by automatically iterating over all exercises.
        """

        # -------------------------------------------
        # Set automatically by investigating program
        # -------------------------------------------

        # Check whether the mode is WEEKLY
        min_reps, max_reps = [], []
        for day in self.days:
            for dynamic_ex in day.dynamic_exercises:
                min_reps.append(dynamic_ex.min_reps)
                max_reps.append(dynamic_ex.max_reps)
        if all_equal(min_reps) and all_equal(max_reps):
            self._min_reps_consistency = 'weekly'
            return None

        # Check if mode is DAILY
        for day in self.days:
            min_reps, max_reps = [], []
            for dynamic_ex in day.dynamic_exercises:
                min_reps.append(dynamic_ex.min_reps)
                max_reps.append(dynamic_ex.max_reps)
            if not all_equal(min_reps) or not all_equal(max_reps):
                self._min_reps_consistency = 'exercise'
                return None
        self._min_reps_consistency = 'daily'

        # -------------------------------------------
        # Respect user wishes if possible
        # -------------------------------------------

        # Set the minimum consistency mode of the program
        if self.min_reps_consistency is not None:

            # Make sure the user defined consistency mode is
            # never more broad than what is allowed by inputs
            if (self._min_reps_consistency == 'exercise' and
                        self.min_reps_consistency != 'exercise'):
                raise ProgramError("Error with 'min_reps_consistency'.")

            if (self._min_reps_consistency == 'daily' and
                        self.min_reps_consistency == 'weekly'):
                raise ProgramError("Error with 'min_reps_consistency'.")

            self._min_reps_consistency = self.min_reps_consistency