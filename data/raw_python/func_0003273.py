async def startlisten(self, vhost = None):
        '''
        Start listen on current servers
        
        :param vhost: return only servers of vhost if specified. '' to return only default servers.
                      None for all servers.
        '''
        servers = self.getservers(vhost)
        for s in servers:
            await s.startlisten()
        return len(servers)