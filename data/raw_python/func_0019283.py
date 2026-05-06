def verify(self):
        """Raise an |ValueError| it the different time grids are
        inconsistent."""
        self.init.verify()
        self.sim.verify()
        if self.init.firstdate > self.sim.firstdate:
            raise ValueError(
                f'The first date of the initialisation period '
                f'({self.init.firstdate}) must not be later '
                f'than the first date of the simulation period '
                f'({self.sim.firstdate}).')
        elif self.init.lastdate < self.sim.lastdate:
            raise ValueError(
                f'The last date of the initialisation period '
                f'({self.init.lastdate}) must not be earlier '
                f'than the last date of the simulation period '
                f'({self.sim.lastdate}).')
        elif self.init.stepsize != self.sim.stepsize:
            raise ValueError(
                f'The initialization stepsize ({self.init.stepsize}) '
                f'must be identical with the simulation stepsize '
                f'({self.sim.stepsize}).')
        else:
            try:
                self.init[self.sim.firstdate]
            except ValueError:
                raise ValueError(
                    'The simulation time grid is not properly '
                    'alligned on the initialization time grid.')