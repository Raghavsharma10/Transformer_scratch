def _initialize_render_dictionary(self):
        """Initialize a dictionary for rendered values.
    
    
        Examples
        -------
        >>> program = Program('My training program')
        >>> program._initialize_render_dictionary()
        >>> program._rendered is False
        False
        """

        self._rendered = dict()

        # Iterate over all weeks
        for week in range(1, self.duration + 1):
            self._rendered[week] = dict()

            # Iterate over all days
            for day in self.days:
                self._rendered[week][day] = dict()

                # Iterate over all main exercises
                for dynamic_ex in day.dynamic_exercises:
                    self._rendered[week][day][dynamic_ex] = dict()