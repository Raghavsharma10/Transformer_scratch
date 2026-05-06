def notifyAppend(self, queue, force):
        '''
        Internal notify for sub-queues
        
        :returns: If the append is blocked by parent, an EventMatcher is returned, None else.
        '''
        if not force and not self.canAppend():
            self.isWaited = True
            return self._matcher
        if self.parent is not None:
            m = self.parent.notifyAppend(self, force)
            if m is not None:
                return m
        self.totalSize = self.totalSize + 1
        return None