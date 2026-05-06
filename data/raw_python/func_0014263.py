def run(self):
        """
        Launch MD simulation, which may consist of:
        1. Optional minimization
        2. Actual MD simulation, with n steps.

        This method also handles reporters.

        Returns
        -------
        positions, velocities : unit.Quantity([natoms, 3])
            Position, velocity of each atom in the system
        box : unit.Quantity([1, 3])
            Periodic conditions box vectors
        """
        if self.verbose:
            status = '#{}'.format(self.stage_index)
            if self.total_stages is not None:
                status += '/{}'.format(self.total_stages)
            status += ': {}'.format(self.name)
            pieces = []
            if self.restrained_atoms is not None:
                pieces.append('restrained {}'.format(self.restrained_atoms))
            if self.constrained_atoms is not None:
                pieces.append('constrained {}'.format(self.constrained_atoms))
            if self.distance_restrained_atoms is not None:
                pieces.append('distance restrained for {} atom pairs'.format(len(self.distance_restrained_atoms)))
            if pieces:
                status += ' [{}]'.format(', '.join(pieces))
            logger.info(status)

        # Add forces
        self.apply_restraints()
        self.apply_constraints()

        if self.barostat:
            self.apply_barostat()

        if self.minimization:
            if self.verbose:
                logger.info('  Minimizing...')
            self.minimize()

        uses_pbc = self.system.usesPeriodicBoundaryConditions()
        if self.steps:
            # Stdout progress
            if self.report and self.progress_reporter not in self.simulation.reporters:
                self.simulation.reporters.append(self.progress_reporter)

            # Log report
            if self.report and self.log_reporter not in self.simulation.reporters:
                self.simulation.reporters.append(self.log_reporter)

            # Trajectory / movie files
            if self.trajectory and self.trajectory_reporter not in self.simulation.reporters:
                self.simulation.reporters.append(self.trajectory_reporter)

            # Checkpoint or restart files
            if self.restart and self.restart_reporter not in self.simulation.reporters:
                self.simulation.reporters.append(self.restart_reporter)

            # MD simulation
            if self.verbose:
                pbc = 'PBC ' if uses_pbc else ''
                conditions = 'NPT' if self.barostat else 'NVT'
                logger.info('  Running {}MD for {} steps @ {}K, {}'.format(pbc, self.steps,
                                                                     self.temperature,
                                                                     conditions))

            with self.handle_exceptions():
                self.simulate()

        if self.save_state_at_end:
            path = self.new_filename(suffix='.state')
            self.simulation.saveState(path)

        # Save and return state
        state = self.simulation.context.getState(getPositions=True, getVelocities=True,
                                                 enforcePeriodicBox=uses_pbc)

        return state.getPositions(), state.getVelocities(), state.getPeriodicBoxVectors()