def backup_simulation(self):
        """
        Creates an emergency report run, .state included
        """
        path = self.new_filename(suffix='_emergency.state')
        self.simulation.saveState(path)
        uses_pbc = self.system.usesPeriodicBoundaryConditions()
        state_kw = dict(getPositions=True, getVelocities=True,
                        getForces=True, enforcePeriodicBox=uses_pbc,
                        getParameters=True, getEnergy=True)
        state = self.simulation.context.getState(**state_kw)
        for reporter in self.simulation.reporters:
            if not isinstance(reporter, app.StateDataReporter):
                reporter.report(self.simulation, state)