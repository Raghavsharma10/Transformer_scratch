def getReliableListeners(self):
        """
        Return an iterable of the listeners which have been added to
        this batch processor.
        """
        for rellist in self.store.query(_ReliableListener, _ReliableListener.processor == self):
            yield rellist.listener