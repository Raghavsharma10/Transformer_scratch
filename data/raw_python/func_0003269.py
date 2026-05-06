def setDaemon(self, runnable, isdaemon, noregister = False):
        '''
        If a runnable is a daemon, it will not keep the main loop running. The main loop will end when all alived runnables are daemons.
        '''
        if not noregister and runnable not in self.registerIndex:
            self.register((), runnable)
        if isdaemon:
            self.daemons.add(runnable)
        else:
            self.daemons.discard(runnable)