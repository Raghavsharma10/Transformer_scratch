def invokeRunnable(self):
        """
        Run my runnable, and reschedule or delete myself based on its result.
        Must be run in a transaction.
        """
        runnable = self.runnable
        if runnable is None:
            self.deleteFromStore()
        else:
            try:
                self.running = True
                newTime = runnable.run()
            finally:
                self.running = False
            self._rescheduleFromRun(newTime)