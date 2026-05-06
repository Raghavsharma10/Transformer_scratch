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
            self._initialized = True

        self._steps[0] += self.interval
        positions = state.getPositions()

        # Serialize
        self._out.write(b''.join([b'\nSTARTOFCHUNK\n',
                                  pickle.dumps([self._steps[0], positions._value]),
                                  b'\nENDOFCHUNK\n']))
        if hasattr(self._out, 'flush'):
            self._out.flush()