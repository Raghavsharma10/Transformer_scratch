def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        if not self._initialized:
            self._initial_clock_time = datetime.now()
            self._initial_simulation_time = state.getTime()
            self._initial_steps = simulation.currentStep
            self._initialized = True

        steps = simulation.currentStep
        time = datetime.now() - self._initial_clock_time
        days = time.total_seconds()/86400.0
        ns = (state.getTime()-self._initial_simulation_time).value_in_unit(u.nanosecond)

        margin = ' ' * self.margin
        ns_day = ns/days
        delta = ((self.total_steps-steps)*time.total_seconds())/steps
        # remove microseconds to have cleaner output
        remaining = timedelta(seconds=int(delta))
        percentage = 100.0*steps/self.total_steps
        if ns_day:
            template = '{}{}/{} steps ({:.1f}%) - {} left @ {:.1f} ns/day                    \r'
        else:
            template = '{}{}/{} steps ({:.1f}%)                                              \r'
        report = template.format(margin, steps, self.total_steps, percentage, remaining, ns_day)
        self._out.write(report)
        if hasattr(self._out, 'flush'):
            self._out.flush()