def append(self, event, force = False):
        '''
        Append an event to queue. The events are classified and appended to sub-queues
        
        :param event: input event
        
        :param force: if True, the event is appended even if the queue is full
        
        :returns: None if appended successfully, or a matcher to match a QueueCanWriteEvent otherwise
        '''
        if self.tree is None:
            if self.parent is None:
                raise IndexError('The queue is removed')
            else:
                return self.parent.parent.append(event, force)
        q = self.tree.matchfirst(event)
        return q.append(event, force)