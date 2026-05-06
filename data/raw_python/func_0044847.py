def _compute_one_step(self, t, fields, pars):
        """
        Compute one step of the simulation, then update the timers.
        """
        fields, pars = self._hook(t, fields, pars)
        self.dt = (self.tmax - t
                   if self.tmax and (t + self.dt >= self.tmax)
                   else self.dt)
        before_compute = time.process_time()
        t, fields = self._scheme(t, fields, self.dt,
                                 pars, hook=self._hook)
        after_compute = time.process_time()
        self._last_running = after_compute - before_compute
        self._total_running += self._last_running
        self._last_timestamp = self._actual_timestamp
        self._actual_timestamp = pendulum.now()
        return t, fields, pars