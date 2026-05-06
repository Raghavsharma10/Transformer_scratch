def compute(self):
        """Generator which yield the actual state of the system every dt.

        Yields
        ------
        tuple : t, fields
            Actual time and updated fields container.
        """
        fields = self.fields
        t = self.t
        pars = self.parameters
        self._started_timestamp = pendulum.now()
        self.stream.emit(self)

        try:
            while True:
                t, fields, pars = self._compute_one_step(t, fields, pars)

                self.i += 1
                self.t = t
                self.fields = fields
                self.parameters = pars
                for pprocess in self.post_processes:
                    pprocess.function(self)
                self.stream.emit(self)
                yield self.t, self.fields

                if self.tmax and (isclose(self.t, self.tmax)):
                    self._end_simulation()
                    return

        except RuntimeError:
            self.status = 'failed'
            raise