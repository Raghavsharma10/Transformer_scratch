def _startReapingProcesses(self):
        """
        Start a LoopingCall that calls reapAllProcesses.
        """
        lc = LoopingCall(self._reapAllProcesses)
        lc.clock = self._reactor
        lc.start(0.1, False)