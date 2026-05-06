def _yield_week_day_dynamic(self):
        """A helper function to reduce the number of nested loops.

        Yields
        -------
        tuple
            A tuple with (week, day, dynamic_ex).

        """
        # Iterate over all weeks
        for week in range(1, self.duration + 1):
            # Iterate over all days
            for day in self.days:
                # Iterate over all main exercises
                for dynamic_ex in day.dynamic_exercises:
                    yield (week, day, dynamic_ex)