async def startlisten(self, connmark = -1):
        '''
        Can call without delegate
        '''
        if connmark is None:
            connmark = self.connmark
        self.scheduler.emergesend(ConnectionControlEvent(self, ConnectionControlEvent.STARTLISTEN, True, connmark))