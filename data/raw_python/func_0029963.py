def state(self, state):
        """Set the current build state and record the time to maintain history.

        Note! This is different from the dataset state. Setting the build set is commiteed to the
        progress table/database immediately. The dstate is also set, but is not committed until the
        bundle is committed. So, the dstate changes more slowly.
        """

        assert state != 'build_bundle'

        self.buildstate.state.current = state
        self.buildstate.state[state] = time()
        self.buildstate.state.lasttime = time()

        self.buildstate.state.error = False
        self.buildstate.state.exception = None
        self.buildstate.state.exception_type = None
        self.buildstate.commit()

        if state in (self.STATES.NEW, self.STATES.CLEANED, self.STATES.BUILT, self.STATES.FINALIZED,
                     self.STATES.SOURCE):
            state = state if state != self.STATES.CLEANED else self.STATES.NEW
            self.dstate = state