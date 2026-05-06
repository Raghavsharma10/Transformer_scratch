def getservers(self, vhost = None):
        '''
        Return current servers
        
        :param vhost: return only servers of vhost if specified. '' to return only default servers.
                      None for all servers.
        '''
        if vhost is not None:
            return [s for s in self.connections if s.protocol.vhost == vhost]
        else:
            return list(self.connections)