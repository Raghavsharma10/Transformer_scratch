def unblockall(self):
        '''
        Remove all blocks from the queue and all sub-queues
        '''
        for q in self.queues.values():
            q.unblockall()
        self.blockEvents.clear()