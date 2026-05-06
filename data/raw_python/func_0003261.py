def register(self, matchers, runnable):
        '''
        Register an iterator(runnable) to scheduler and wait for events
        
        :param matchers: sequence of EventMatchers
        
        :param runnable: an iterator that accept send method
        
        :param daemon: if True, the runnable will be registered as a daemon.
        '''
        if getattr(self, 'syscallfunc', None) is not None and getattr(self, 'syscallrunnable', None) is None:
            # Inject this register
            self.syscallrunnable = runnable
        else:
            for m in matchers:
                self.matchtree.insert(m, runnable)
                events = self.eventtree.findAndRemove(m)
                for e in events:
                    self.queue.unblock(e)
                if m.indices[0] == PollEvent._classname0 and len(m.indices) >= 2:
                    self.polling.onmatch(m.indices[1], None if len(m.indices) <= 2 else m.indices[2], True)
            self.registerIndex.setdefault(runnable, set()).update(matchers)