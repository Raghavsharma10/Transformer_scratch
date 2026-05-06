async def reset(self, force = True, connmark = None):
        '''
        Can call without delegate
        '''
        if connmark is None:
            connmark = self.connmark
        self.scheduler.emergesend(ConnectionControlEvent(self, ConnectionControlEvent.RESET, force, connmark))