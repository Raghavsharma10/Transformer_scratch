def removeReliableListener(self, listener):
        """
        Remove a previously added listener.
        """
        self.store.query(_ReliableListener,
                         attributes.AND(_ReliableListener.processor == self,
                                        _ReliableListener.listener == listener)).deleteFromStore()
        self.store.query(BatchProcessingError,
                         attributes.AND(BatchProcessingError.processor == self,
                                        BatchProcessingError.listener == listener)).deleteFromStore()