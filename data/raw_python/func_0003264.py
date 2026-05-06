def ignore(self, matcher):
        '''
        Unblock and ignore the matched events, if any. 
        '''
        events  = self.eventtree.findAndRemove(matcher)
        for e in events:
            self.queue.unblock(e)
            e.canignore = True