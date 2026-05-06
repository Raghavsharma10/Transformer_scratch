async def stoplisten(self, connmark = -1):
        '''
        Can call without delegate
        '''
        if connmark is None:
            connmark = self.connmark
        self.scheduler.emergesend(ConnectionControlEvent(self, ConnectionControlEvent.STOPLISTEN, True, connmark))