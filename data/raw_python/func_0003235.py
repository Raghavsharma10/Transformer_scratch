async def reconnect(self, force = True, connmark = None):
        '''
        Can call without delegate
        '''
        if connmark is None:
            connmark = self.connmark
        self.scheduler.emergesend(ConnectionControlEvent(self, ConnectionControlEvent.RECONNECT, force, connmark))