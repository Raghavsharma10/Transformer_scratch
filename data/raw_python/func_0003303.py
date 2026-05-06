def removeSubQueue(self, queue):
        '''
        remove a sub queue from current queue.
        
        This unblock the sub-queue, retrieve all events from the queue and put them back to the parent.
        
        Call clear on the sub-queue first if the events are not needed any more.
        
        :param queue: the name or queue object to remove
        
        :returns: ((queueevents,...), (queueEmptyEvents,...)) Possible queue events from removing sub-queues
        '''
        q = self.queueindex[queue]
        q[1].unblockall()
        q[1]._removeFromTree()
        ret = ([],[])
        while q[1].canPop():
            r = q[1].pop()
            self.append(r[0], True)
            ret[0].extend(r[1])
            ret[1].extend(r[2])
        self.queues[q[0]].removeSubQueue(q[1])
        # Remove from index
        if q[2] is not None:
            del self.queueindex[q[2]]
        del self.queueindex[q[1]]
        newblocked =  not self.canPop()
        if newblocked != self.blocked:
            self.blocked = newblocked
            if self.parent is not None:
                self.parent.notifyBlock(self, newblocked)
        return ret