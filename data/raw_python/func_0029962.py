def error_state(self):
        """Set the error condition"""
        self.buildstate.state.lasttime = time()
        self.buildstate.commit()
        return self.buildstate.state.error