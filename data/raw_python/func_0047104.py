def _validate(self):
        """
        The purpose of this method is to verify that the user has set sensible
        values for the training program before rendering. The user will still
        be able to render, but error messages will be printed. This method:
            
            * Validates that the average intensity is in the range [65, 85].
            * Validates that the number of repetitions is in the range [15, 45].
            * Validates that 'reps_to_intensity_func' maps to [0, 100].
            * Validates that 'reps_to_intensity_func' is a decreasing function.
            * Validates that the exercises do not grow more than 2.5% per week.
            
        Apart from these sanity checks, the user is on his own.
        """
        # Validate the intensity
        if max([s * self.intensity for s in self._intensity_scalers]) > 85:
            warnings.warn('\nWARNING: Average intensity is > 85.')

        if min([s * self.intensity for s in self._intensity_scalers]) < 65:
            warnings.warn('\nWARNING: Average intensity is < 65.')

        # Validate the repetitions
        if max([s * self.reps_per_exercise for s in self._rep_scalers]) > 45:
            warnings.warn('\nWARNING: Number of repetitions > 45.')

        if min([s * self.reps_per_exercise for s in self._rep_scalers]) < 15:
            warnings.warn('\nWARNING: Number of repetitions < 15.')

        # Validate the 'reps_to_intensity_func'
        for x1, x2 in zip(range(1, 20), range(2, 21)):
            y1 = self.reps_to_intensity_func(x1)
            y2 = self.reps_to_intensity_func(x2)
            if y1 < y2:
                warnings.warn("\n'reps_to_intensity_func' is not decreasing.")

        if any(self.reps_to_intensity_func(x) > 100 for x in range(1, 20)):
            warnings.warn("\n'reps_to_intensity_func' maps to > 100.")

        if any(self.reps_to_intensity_func(x) < 0 for x in range(1, 20)):
            warnings.warn("\n'reps_to_intensity_func' maps to < 0.")

        # Validate the exercises
        for day in self.days:
            for dynamic_ex in day.dynamic_exercises:
                start, end = dynamic_ex.start_weight, dynamic_ex.final_weight
                percentage_growth = (end / start) ** (1 / self.duration)
                percentage_growth = dynamic_ex.weekly_growth(self.duration)
                if percentage_growth > 4:
                    msg = '\n"{}" grows with {}% each week.'.format(
                        dynamic_ex.name, percentage_growth)
                    warnings.warn(msg)