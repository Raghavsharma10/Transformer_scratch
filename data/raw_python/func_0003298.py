def _pop(self):
        '''
        Actual pop
        '''
        if not self.canPop():
            raise IndexError('pop from an empty or blocked queue')
        priority = self.prioritySet[-1]
        ret = self.queues[priority]._pop()
        self.outputStat = self.outputStat + 1
        self.totalSize = self.totalSize - 1
        if self.isWaited and self.canAppend():
            self.isWaited = False
            ret[1].append(QueueCanWriteEvent(self))
        if self.isWaitEmpty and not self:
            self.isWaitEmpty = False
            ret[2].append(QueueIsEmptyEvent(self))
        return ret