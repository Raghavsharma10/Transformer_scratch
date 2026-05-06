async def subscribe(self, container, *keys):
        '''
        Subscribe to specified channels
        
        :param container: routine container
        
        :param *keys: subscribed channels
        
        :returns: list of event matchers for the specified channels
        '''
        await self._get_subscribe_connection(container)
        realkeys = []
        for k in keys:
            count = self._subscribecounter.get(k, 0)
            if count == 0:
                realkeys.append(k)
            self._subscribecounter[k] = count + 1
        if realkeys:
            await self._protocol.execute_command(self._subscribeconn, container, 'SUBSCRIBE', *realkeys)
        return [self._protocol.subscribematcher(self._subscribeconn, k, None, RedisSubscribeMessageEvent.MESSAGE) for k in keys]