def unblockqueue(self, queue):
        '''
        Remove blocked events from the queue and all subqueues. Usually used after queue clear/unblockall to prevent leak.
        
        :returns: the cleared events
        '''
        subqueues = set()
        def allSubqueues(q):
            subqueues.add(q)
            subqueues.add(q.defaultQueue)
            for v in q.queueindex.values():
                if len(v) == 3:
                    allSubqueues(v[1])
        allSubqueues(queue)
        events = [k for k,v in self.blockEvents.items() if v in subqueues]
        for e in events:
            del self.blockEvents[e]
        return events