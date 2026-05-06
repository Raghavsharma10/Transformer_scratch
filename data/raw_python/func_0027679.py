def mark(self):
        """
        Mark the unit of work as failed in the database and update the listener
        so as to skip it next time.
        """
        self.reliableListener.lastRun = extime.Time()
        BatchProcessingError(
            store=self.reliableListener.store,
            processor=self.reliableListener.processor,
            listener=self.reliableListener.listener,
            item=self.workUnit,
            error=self.failure.getErrorMessage())