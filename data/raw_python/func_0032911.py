def stopReceivingBoxes(self, reason):
        """
        Stop observing log events.
        """
        AMP.stopReceivingBoxes(self, reason)
        log.removeObserver(self._emit)