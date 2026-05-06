def registerAllHandlers(self, handlerDict):
        '''
        Register self to scheduler
        '''
        self.handlers.update(handlerDict)
        if hasattr(handlerDict, 'keys'):
            self.scheduler.register(handlerDict.keys(), self)
        else:
            self.scheduler.register(tuple(h[0] for h in handlerDict), self)
        self._setDaemon()