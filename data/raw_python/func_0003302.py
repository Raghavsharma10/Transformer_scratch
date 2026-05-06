def addSubQueue(self, priority, matcher, name = None, maxdefault = None, maxtotal = None, defaultQueueClass = FifoQueue):
        '''
        add a sub queue to current queue, with a priority and a matcher
        
        :param priority: priority of this queue. Larger is higher, 0 is lowest.
        
        :param matcher: an event matcher to catch events. Every event match the criteria will be stored in this queue.
        
        :param name: a unique name to identify the sub-queue. If none, the queue is anonymous. It can be any hashable value.
        
        :param maxdefault: max length for default queue.
        
        :param maxtotal: max length for sub-queue total, including sub-queues of sub-queue
        '''
        if name is not None and name in self.queueindex:
            raise IndexError("Duplicated sub-queue name '" + str(name) + "'")
        subtree = self.tree.subtree(matcher, True)
        newPriority = self.queues.setdefault(priority, CBQueue.MultiQueue(self, priority))
        newQueue = CBQueue(subtree, newPriority, maxdefault, maxtotal, defaultQueueClass)
        newPriority.addSubQueue(newQueue)
        qi = [priority, newQueue, name]
        if name is not None:
            self.queueindex[name] = qi
        self.queueindex[newQueue] = qi
        return newQueue