def unregisterall(self, runnable):
        '''
        Unregister all matches and detach the runnable. Automatically called when runnable returns StopIteration.
        '''
        if runnable in self.registerIndex:
            for m in self.registerIndex[runnable]:
                self.matchtree.remove(m, runnable)
                if m.indices[0] == PollEvent._classname0 and len(m.indices) >= 2:
                    self.polling.onmatch(m.indices[1], None if len(m.indices) <= 2 else m.indices[2], False)
            del self.registerIndex[runnable]
            self.daemons.discard(runnable)