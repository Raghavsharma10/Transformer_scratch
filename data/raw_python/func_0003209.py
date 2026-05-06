async def execute_command(self, container, *args):
        '''
        Execute command on Redis server:
          - For (P)SUBSCRIBE/(P)UNSUBSCRIBE, the command is sent to the subscribe connection.
            It is recommended to use (p)subscribe/(p)unsubscribe method instead of directly call the command
          - For BLPOP, BRPOP, BRPOPLPUSH, the command is sent to a separated connection. The connection is
            recycled after command returns.
          - For other commands, the command is sent to the default connection.
        '''
        if args:
            cmd = _str(args[0]).upper()
            if cmd in ('SUBSCRIBE', 'UNSUBSCRIBE', 'PSUBSCRIBE', 'PUNSUBSCRIBE'):
                await self._get_subscribe_connection(container)
                return await self._protocol.execute_command(self._subscribeconn, container, *args)
            elif cmd in ('BLPOP', 'BRPOP', 'BRPOPLPUSH'):
                c = await self.get_connection(container)
                with c.context(container):
                    return await c.execute_command(container, *args)
        return await RedisClientBase.execute_command(self, container, *args)