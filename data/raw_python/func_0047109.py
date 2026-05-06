def _set_min_reps(self):
        """Populate the _rendered dictionary with entries corresponding
        to the minimum number of reps to go to for each exercise
        and for the entire duration.
    
    
        Examples
        -------
        >>> program = Program('My training program', duration = 2)
        >>> bench_press = DynamicExercise('Bench', 100, 120)
        >>> day = Day(exercises = [bench_press])
        >>> program.add_days(day)
        >>> program._initialize_render_dictionary()
        >>> program._autoset_min_reps_consistency()
        >>> program._set_min_reps()
        >>> for week in range(1, program.duration + 1):
        ...     for day in program.days:
        ...         for d_ex in day.dynamic_exercises:
        ...             print(program._rendered[week][day][d_ex]['minimum'] > 0)
        True
        True
        """

        min_percent = self.minimum_percentile
        # --------------------------------
        # If the mode is weekly, set minimum reps on a weekly basis
        # --------------------------------
        if self._min_reps_consistency == 'weekly':

            # Set up generator. Only one is needed
            exercise = self.days[0].dynamic_exercises[0]
            margs = exercise.min_reps, exercise.max_reps, min_percent
            low, high = min_between(*margs)
            generator = RepellentGenerator(list(range(low, high + 1)))

            # Use generator to populate the dictionary with minimum values
            for week in range(1, self.duration + 1):
                min_rep_week = generator.generate_one()
                for day in self.days:
                    for d_ex in day.dynamic_exercises:
                        self._rendered[week][day][d_ex]['minimum'] = min_rep_week

        # --------------------------------
        # If the mode is daily, set minimum reps on a daily basis
        # --------------------------------
        if self._min_reps_consistency == 'daily':

            # Set up generators. One is needed for each day
            generators = dict()
            for day in self.days:
                exercise = day.dynamic_exercises[0]
                margs = exercise.min_reps, exercise.max_reps, min_percent
                low, high = min_between(*margs)
                generator = RepellentGenerator(list(range(low, high + 1)))
                generators[day] = generator

            # Use generators to populate the dictionary with minimum values
            for week in range(1, self.duration + 1):
                for day in self.days:
                    min_rep_day = generators[day].generate_one()
                    for d_ex in day.dynamic_exercises:
                        self._rendered[week][day][d_ex]['minimum'] = min_rep_day

        # --------------------------------
        # If the mode is by exercise, set minimum reps on an exercise basis
        # --------------------------------           
        if self._min_reps_consistency == 'exercise':

            # Set up generators. One is needed for each exercise
            generators = dict()
            for day in self.days:
                for d_ex in day.dynamic_exercises:
                    margs = d_ex.min_reps, d_ex.max_reps, min_percent
                    low, high = min_between(*margs)
                    generator = RepellentGenerator(list(range(low, high + 1)))
                    generators[d_ex] = generator

            # Use generators to populate the dictionary with minimum values
            for week in range(1, self.duration + 1):
                for day in self.days:
                    for d_ex in day.dynamic_exercises:
                        min_rep_ex = generators[d_ex].generate_one()
                        self._rendered[week][day][d_ex]['minimum'] = min_rep_ex