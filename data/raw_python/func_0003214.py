def subscribe_state_matcher(self, container, connected = True):
        '''
        Return a matcher to match the subscribe connection status.
        
        :param container: a routine container. NOTICE: this method is not a routine.
        
        :param connected: if True, the matcher matches connection up. If False, the matcher matches
               connection down.
        
        :returns: an event matcher.
        '''
        if not self._subscribeconn:
            self._subscribeconn = self._create_client(container)
        return RedisConnectionStateEvent.createMatcher(
                    RedisConnectionStateEvent.CONNECTION_UP if connected else RedisConnectionStateEvent.CONNECTION_DOWN,
                    self._subscribeconn
                    )