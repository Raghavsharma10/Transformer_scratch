def set_trace(self, frame=None):
        """Start debugging from `frame`.

        If frame is not specified, debugging starts from caller's frame.
        """
        # First disable tracing temporarily as set_trace() may be called while
        # tracing is in use. For example when called from a signal handler and
        # within a debugging session started with runcall().
        self.settrace(False)

        if not frame:
            frame = sys._getframe().f_back
        frame.f_trace = self.trace_dispatch

        # Do not change botframe when the debuggee has been started from an
        # instance of Pdb with one of the family of run methods.
        self.reset(ignore_first_call_event=False, botframe=self.botframe)
        self.topframe = frame
        while frame:
            if frame is self.botframe:
                break
            botframe = frame
            frame = frame.f_back
        else:
            self.botframe = botframe

        # Must trace the bottom frame to disable tracing on termination,
        # see issue 13044.
        if not self.botframe.f_trace:
            self.botframe.f_trace = self.trace_dispatch

        self.settrace(True)