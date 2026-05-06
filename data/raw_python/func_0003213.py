async def shutdown(self, container, force=False):
        '''
        Shutdown all connections. Exclusive connections created by get_connection will shutdown after release()
        '''
        p = self._connpool
        self._connpool = []
        self._shutdown = True
        if self._defaultconn:
            p.append(self._defaultconn)
            self._defaultconn = None
        if self._subscribeconn:
            p.append(self._subscribeconn)
            self._subscribeconn = None
        await container.execute_all([self._shutdown_conn(container, o, force)
                                       for o in p])