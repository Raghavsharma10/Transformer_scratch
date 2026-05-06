def setPriority(self, queue, priority):
        '''
        Set priority of a sub-queue
        '''
        q = self.queueindex[queue]
        self.queues[q[0]].removeSubQueue(q[1])
        newPriority = self.queues.setdefault(priority, CBQueue.MultiQueue(self, priority))
        q[0] = priority
        newPriority.addSubQueue(q[1])