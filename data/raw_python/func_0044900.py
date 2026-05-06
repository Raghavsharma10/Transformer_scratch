def cleanup(self):
        """Used internally.

        Cleans up the sched references in the proactor. If you use this don't use
        it while the :class:`Scheduler` (:func:`run`) is still running.
        """

        if hasattr(self, 'proactor'):
            if hasattr(self.proactor, 'scheduler'):
                del self.proactor.scheduler
            if hasattr(self.proactor, 'close'):
                self.proactor.close()