def unregister(self, matchers, runnable):
        '''
        Unregister an iterator(runnable) and stop waiting for events
        
        :param matchers: sequence of EventMatchers
        
        :param runnable: an iterator that accept send method
        '''
        for m in matchers:
            self.matchtree.remove(m, runnable)
            if m.indices[0] == PollEvent._classname0 and len(m.indices) >= 2:
                self.polling.onmatch(m.indices[1], None if len(m.indices) <= 2 else m.indices[2], False)
        self.registerIndex.setdefault(runnable, set()).difference_update(matchers)