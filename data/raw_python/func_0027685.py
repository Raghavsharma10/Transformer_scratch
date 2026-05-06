def itemAdded(self):
        """
        Called to indicate that a new item of the type monitored by this batch
        processor is being added to the database.

        If this processor is not already scheduled to run, this will schedule
        it.  It will also start the batch process if it is not yet running and
        there are any registered remote listeners.
        """
        localCount = self.store.query(
            _ReliableListener,
            attributes.AND(_ReliableListener.processor == self,
                           _ReliableListener.style == iaxiom.LOCAL),
            limit=1).count()

        remoteCount = self.store.query(
            _ReliableListener,
            attributes.AND(_ReliableListener.processor == self,
                           _ReliableListener.style == iaxiom.REMOTE),
            limit=1).count()

        if localCount and self.scheduled is None:
            self.scheduled = extime.Time()
            iaxiom.IScheduler(self.store).schedule(self, self.scheduled)
        if remoteCount:
            batchService = iaxiom.IBatchService(self.store, None)
            if batchService is not None:
                batchService.start()