def block(self, event, emptyEvents = ()):
        '''
        Return a recently popped event to queue, and block all later events until unblock.
        
        Only the sub-queue directly containing the event is blocked, so events in other queues may still be processed.
        It is illegal to call block and unblock in different queues with a same event.
        
        :param event: the returned event. When the queue is unblocked later, this event will be popped again.
        
        :param emptyEvents: reactivate the QueueIsEmptyEvents
        '''
        q = self.tree.matchfirst(event)
        q.block(event)
        self.blockEvents[event] = q
        for ee in emptyEvents:
            ee.queue.waitForEmpty()