def add_exercises(self, *exercises):
        """Add the exercises to the day. The method will automatically infer
        whether a static or dynamic exercise is passed to it.

        Parameters
        ----------
        *exercises
            An unpacked tuple of exercises.


        Examples
        -------
        >>> monday = Day(name = 'Monday')
        >>> curls = StaticExercise('Curls', '3 x 12')
        >>> pulldowns = StaticExercise('Pulldowns', '4 x 10')
        >>> monday.add_exercises(curls, pulldowns)
        >>> curls in monday.static_exercises
        True
        >>> pulldowns in monday.static_exercises
        True
        """
        for exercise in list(exercises):
            if isinstance(exercise, DynamicExercise):
                self.dynamic_exercises.append(exercise)

            if isinstance(exercise, StaticExercise):
                self.static_exercises.append(exercise)