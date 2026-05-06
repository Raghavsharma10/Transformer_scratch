async def psubscribe(self, container, *keys):
        '''
        Subscribe to specified globs
        
        :param container: routine container
        
        :param \*keys: subscribed globs
        
        :returns: list of event matchers for the specified globs 
        '''
        await self._get_subscribe_connection(container)
        realkeys = []
        for k in keys:
            count = self._psubscribecounter.get(k, 0)
            if count == 0:
                realkeys.append(k)
            self._psubscribecounter[k] = count + 1
        await self._protocol.execute_command(self._subscribeconn, container, 'PSUBSCRIBE', *realkeys)
        return [self._protocol.subscribematcher(self._subscribeconn, k, None, RedisSubscribeMessageEvent.PMESSAGE) for k in keys]