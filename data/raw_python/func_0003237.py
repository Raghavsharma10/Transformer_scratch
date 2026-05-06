async def write(self, event, ignoreException = True):
        '''
        Can call without delegate
        '''
        connmark = self.connmark
        if self.connected:
            def _until():
                if not self.connected or self.connmark != connmark:
                    return True
            r = await self.wait_for_send(event, until=_until)
            if r:
                if ignoreException:
                    return
                else:
                    raise
        else:
            if not ignoreException:
                raise ConnectionResetException