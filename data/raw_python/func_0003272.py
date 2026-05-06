async def stoplisten(self, vhost = None):
        '''
        Stop listen on current servers
        
        :param vhost: return only servers of vhost if specified. '' to return only default servers.
                      None for all servers.
        '''
        servers = self.getservers(vhost)
        for s in servers:
            await s.stoplisten()
        return len(servers)