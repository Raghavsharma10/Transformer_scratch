def create(self):
        """
        Create the subqueue to change the default behavior of Lock to semaphore.
        """
        self.queue = self.scheduler.queue.addSubQueue(self.priority, LockEvent.createMatcher(self.context, self.key),
                                         maxdefault = self.size, defaultQueueClass = CBQueue.AutoClassQueue.initHelper('locker', subqueuelimit = 1))