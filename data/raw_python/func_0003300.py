def _clear(self):
        '''
        Actual clear
        '''
        ret = ([],[])
        for q in self.queues.values():
            pr = q._clear()
            ret[0].extend(pr[0])
            ret[1].extend(pr[1])
        self.totalSize = 0
        del self.prioritySet[:]
        if self.isWaited and self.canAppend():
            self.isWaited = False
            ret[0].append(QueueCanWriteEvent(self))
        if self.isWaitEmpty and not self:
            self.isWaitEmpty = False
            ret[1].append(QueueIsEmptyEvent(self))
        self.blockEvents.clear()
        return ret