def _set_trace(self, skip=0):
        """Start debugging from here."""
        frame = sys._getframe().f_back
        # go up the specified number of frames
        for i in range(skip):
            frame = frame.f_back
        self.reset()
        while frame:
            frame.f_trace = self.trace_dispatch
            self.botframe = frame
            frame = frame.f_back
        self.set_step()
        sys.settrace(self.trace_dispatch)