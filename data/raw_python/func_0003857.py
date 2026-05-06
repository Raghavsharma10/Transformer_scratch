def parents(self):
        """
        Returns an simple FIFO queue with the ancestors and itself.
        """
        q = self.__parent__.parents()
        q.put(self)
        return q