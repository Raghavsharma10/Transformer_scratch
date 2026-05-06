def render(self, validate=True):
        """Render the training program to perform the calculations.
        The program can be rendered several times to produce new
        information given the same input parameters.
    
        Parameters
        ----------
        validate
            Boolean that indicates whether or not to run a validation
            heurestic on the program before rendering. The validation
            will warn the user if inputs seem unreasonable.

        """

        # --------------------------------
        # Prepare for rendering the dynamic exercises
        # --------------------------------

        # Set the minimum repetitions consistency mode,
        # which is either 'weekly', 'daily' or 'exercise'
        self._autoset_min_reps_consistency()

        # Initialize the structure of the _rendered dictionary
        self._initialize_render_dictionary()

        # Set the day names
        for i, day in enumerate(self.days):
            day.name = prioritized_not_None(day.name, 'Day {}'.format(i + 1))

        # Set the minimum reps per week in the render dictionary
        self._set_min_reps()

        # Set the scalers
        self._set_scalers()

        # Validate the program if the user wishes to validate
        if validate:
            self._validate()

        # --------------------------------
        # Render the dynamic exercises
        # --------------------------------

        for (week, day, dyn_ex) in self._yield_week_day_dynamic():

            # The minimum repeition to work up to
            min_rep = self._rendered[week][day][dyn_ex]['minimum']

            # The desired repetitions to work up to
            local_r, global_r = dyn_ex.reps, self.reps_per_exercise
            total_reps = prioritized_not_None(local_r, global_r)
            desired_reps = total_reps * self._rep_scalers[week - 1]
            self._rendered[week][day][dyn_ex]['desired_reps'] = int(
                desired_reps)

            # The desired intensity to work up to
            local_i, global_i = dyn_ex.intensity, self.intensity
            intensity_unscaled = prioritized_not_None(local_i, global_i)
            scale_factor = self._intensity_scalers[week - 1]
            desired_intensity = intensity_unscaled * scale_factor
            self._rendered[week][day][dyn_ex]['desired_intensity'] = int(desired_intensity)

            # A dictionary is returned with keys 'reps' and 'intensities'
            render_args = dyn_ex, min_rep, desired_reps, desired_intensity, validate
            out = self._render_dynamic(*render_args)

            # Calculate the 1RM at this point in time
            start_w, final_w = dyn_ex.start_weight, dyn_ex.final_weight
            args = (week, start_w, final_w, 1, self.duration)
            weight = self.progression_func(*args)

            # Define a function to prettify the weights
            def pretty_weight(weight, i, round_function):
                weight = round_function(weight * i / 100)
                if weight % 1 == 0:
                    return int(weight)
                return weight

            # Use the local rounding function if available,
            # if not use the global rounding function
            round_func = prioritized_not_None(dyn_ex.round, self.round)

            # Create pretty strings
            tuple_generator = zip(out['intensities'], out['reps'])
            pretty_gen = ((str(r), str(pretty_weight(weight, i, round_func)) +
                           self.units) for (i, r) in tuple_generator)
            joined_gen = (self.REP_SET_SEP.join(list(k)) for k in pretty_gen)

            out['strings'] = list(joined_gen)

            # The _rendered dictionary has keys
            # ['minimum', 'desired_reps', 'desired_intensity'].
            # Update with the ['intensities', 'reps', 'strings'] keys
            self._rendered[week][day][dyn_ex].update(out)