def registerHandler(self, matcher, handler):
        '''
        Register self to scheduler
        '''
        self.handlers[matcher] = handler
        self.scheduler.register((matcher,), self)
        self._setDaemon()