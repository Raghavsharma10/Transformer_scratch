def startReceivingBoxes(self, sender):
        """
        Start observing log events for stat events to send.
        """
        AMP.startReceivingBoxes(self, sender)
        log.addObserver(self._emit)