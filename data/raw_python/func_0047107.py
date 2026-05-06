def _render_dynamic(self, dynamic_exercise, min_rep,
                        desired_reps, desired_intensity, validate):
        """
        Render a single dynamic exercise.
        This is done for each exercise every week.
        """

        # --------------------------------
        # Generate possible repstring and calculate penalties
        # --------------------------------
        repstrings = []
        for k in range(self.TIMES_TO_RENDER):

            # If going to minimum, add the minimum repetition to the reps
            if self.go_to_min:
                reps = generate_reps(min_rep,
                                     dynamic_exercise.max_reps,
                                     desired_reps - min_rep,
                                     [min_rep])
            else:
                reps = generate_reps(min_rep,
                                     dynamic_exercise.max_reps,
                                     desired_reps)

            # Calculate the penalty
            intensities = list(map(self.reps_to_intensity_func, reps))

            pargs = reps, intensities, desired_reps, desired_intensity, min_rep
            penalty_value = self.repstring_penalty(*pargs)

            repstrings.append((penalty_value, reps, intensities))

        # --------------------------------
        # Find the best generated repstring and verify it
        # --------------------------------
        best_repstring = min(repstrings)
        (penalty_value, reps, intensities) = best_repstring

        # Perform a sanity check:
        # If repetitions are too high, a low average intensity cannot be attained
        if desired_intensity > self.reps_to_intensity_func(min_rep) and validate:
            msg = """
WARNING: The exercise '{}' is restricted to repetitions in the range [{}, {}],
but the desired average intensity for this week is {}. Reaching this intensity
is not attainable since it corresponds to repetitions lower than {}.
SOLUTION: Either (1) allow lower repetitions, (2) change the desired intensity
or (3) ignore this message. The software will do it's best to remedy this.
""".format(dynamic_exercise.name, dynamic_exercise.max_reps, min_rep,
           desired_intensity, min_rep)
            warnings.warn(msg)

        # Perform a sanity check:
        # If repetitions are too low, a high average intensity cannot be attained
        if desired_intensity < self.reps_to_intensity_func(
                dynamic_exercise.max_reps) and validate:
            msg = """
WARNING: The exercise '{}' is restricted to repetitions in the range [{}, {}],
but the desired average intensity for this week is {}. Reaching this intensity
is not attainable since it corresponds to repetitions higher than {}.
SOLUTION: Either (1) allow higher repetitions, (2) change the desired intensity
or (3) ignore this message. The software will do it's best to remedy this.
""".format(dynamic_exercise.name, dynamic_exercise.max_reps, min_rep,
           desired_intensity, dynamic_exercise.max_reps)
            warnings.warn(msg)

        return {'reps': reps, 'intensities': intensities}