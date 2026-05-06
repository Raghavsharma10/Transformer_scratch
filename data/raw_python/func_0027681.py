def addReliableListener(self, listener, style=iaxiom.LOCAL):
        """
        Add the given Item to the set which will be notified of Items
        available for processing.

        Note: Each Item is processed synchronously.  Adding too many
        listeners to a single batch processor will cause the L{step}
        method to block while it sends notification to each listener.

        @param listener: An Item instance which provides a
        C{processItem} method.

        @return: An Item representing L{listener}'s persistent tracking state.
        """
        existing = self.store.findUnique(_ReliableListener,
                                         attributes.AND(_ReliableListener.processor == self,
                                                        _ReliableListener.listener == listener),
                                         default=None)
        if existing is not None:
            return existing

        for work in self.store.query(self.workUnitType,
                                     sort=self.workUnitType.storeID.descending,
                                     limit=1):
            forwardMark = work.storeID
            backwardMark = work.storeID + 1
            break
        else:
            forwardMark = 0
            backwardMark = 0

        if self.scheduled is None:
            self.scheduled = extime.Time()
            iaxiom.IScheduler(self.store).schedule(self, self.scheduled)

        return _ReliableListener(store=self.store,
                                 processor=self,
                                 listener=listener,
                                 forwardMark=forwardMark,
                                 backwardMark=backwardMark,
                                 style=style)