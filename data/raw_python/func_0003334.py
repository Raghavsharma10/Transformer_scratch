def route(self, path, routinemethod, container = None, host = None, vhost = None, method = [b'GET', b'HEAD']):
        '''
        Route specified path to a WSGI-styled routine factory
        
        :param path: path to match, can be a regular expression
         
        :param routinemethod: factory function routinemethod(env), env is an Environment object
                see also utils.http.Environment
        
        :param container: routine container
        
        :param host: if specified, only response to request to specified host
        
        :param vhost: if specified, only response to request to specified vhost.
                      If not specified, response to dispatcher default vhost.
        
        :param method: if specified, response to specified methods
        '''
        self.routeevent(path, statichttp(container)(routinemethod), container, host, vhost, method)