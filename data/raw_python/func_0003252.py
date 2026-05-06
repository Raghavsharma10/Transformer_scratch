async def _dataobject_update_detect(self, _initialkeys, _savedresult):
        """
        Coroutine that wait for retrieved value update notification
        """
        def expr(newvalues, updatedvalues):
            if any(v.getkey() in _initialkeys for v in updatedvalues if v is not None):
                return True
            else:
                return self.shouldupdate(newvalues, updatedvalues)
        while True:
            updatedvalues, _ = await multiwaitif(_savedresult, self, expr, True)
            if not self._updatedset:
                self.scheduler.emergesend(FlowUpdaterNotification(self, FlowUpdaterNotification.DATAUPDATED))
            self._updatedset.update(updatedvalues)