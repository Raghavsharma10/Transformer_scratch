def updateobjects(self, updatedvalues):
        """
        Force a update notification on specified objects, even if they are not actually updated
        in ObjectDB
        """
        if not self._updatedset:
            self.scheduler.emergesend(FlowUpdaterNotification(self, FlowUpdaterNotification.DATAUPDATED))
        self._updatedset.update(set(updatedvalues).intersection(self._savedresult))