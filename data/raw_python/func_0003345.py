async def wait_for_send(self, event, *, until=None):
        '''
        Send an event to the main event queue. Can call without delegate.
        
        :param until: if the callback returns True, stop sending and return
        
        :return: the last True value the callback returns, or None
        '''
        while True:
            if until:
                r = until()
                if r:
                    return r
            waiter = self.scheduler.send(event)
            if waiter is None:
                break
            await waiter