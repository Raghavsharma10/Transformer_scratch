def stop(self, measurementId, failureReason=None):
        """
        informs the target the named measurement has completed
        :param measurementId: the measurement that has completed.
        :return:
        """
        if failureReason is None:
            self.endResponseCode = self._doPut(self.sendURL + "/complete")
        else:
            self.endResponseCode = self._doPut(self.sendURL + "/failed", data={'failureReason': failureReason})
        self.sendURL = None