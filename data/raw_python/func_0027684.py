def getFailedItems(self):
        """
        Return an iterable of two-tuples of listeners which raised an
        exception from C{processItem} and the item which was passed as
        the argument to that method.
        """
        for failed in self.store.query(BatchProcessingError, BatchProcessingError.processor == self):
            yield (failed.listener, failed.item)