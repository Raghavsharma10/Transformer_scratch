async def unsubscribe(self, container, *keys):
        '''
        Unsubscribe specified channels. Every subscribed key should be unsubscribed exactly once, even if duplicated subscribed.
        
        :param container: routine container
        
        :param \*keys: subscribed channels
        '''
        await self._get_subscribe_connection(container)
        realkeys = []
        for k in keys:
            count = self._subscribecounter.get(k, 0)
            if count <= 1:
                realkeys.append(k)
                try:
                    del self._subscribecounter[k]
                except KeyError:
                    pass
            else:
                self._subscribecounter[k] = count - 1
        if realkeys:
            await self._protocol.execute_command(self._subscribeconn, container, 'UNSUBSCRIBE', *realkeys)