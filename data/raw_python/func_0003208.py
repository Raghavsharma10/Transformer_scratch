async def get_connection(self, container):
        '''
        Get an exclusive connection, useful for blocked commands and transactions.
        
        You must call release or shutdown (not recommanded) to return the connection after use.
        
        :param container: routine container
        
        :returns: RedisClientBase object, with some commands same as RedisClient like execute_command,
                  batch_execute, register_script etc.
        '''
        if self._connpool:
            conn = self._connpool.pop()
            return RedisClientBase(conn, self)
        else:
            conn = self._create_client(container)
            await RedisClientBase._get_connection(self, container, conn)
            await self._protocol.send_command(conn, container, 'SELECT', str(self.db))
            return RedisClientBase(conn, self)