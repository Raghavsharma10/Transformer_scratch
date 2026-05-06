def run_timeit(self, stmt, setup):
        """ Create the function call statement as a string used for timeit. """
        _timer = timeit.Timer(stmt=stmt, setup=setup)
        trials = _timer.repeat(self.timeit_repeat, self.timeit_number)
        self.time_average_seconds = sum(trials) / len(trials) / self.timeit_number
        # Convert into reasonable time units
        time_avg = convert_time_units(self.time_average_seconds)
        return time_avg