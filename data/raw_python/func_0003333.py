def routeevent(self, path, routinemethod, container = None, host = None, vhost = None, method = [b'GET', b'HEAD']):
        '''
        Route specified path to a routine factory
        
        :param path: path to match, can be a regular expression
         
        :param routinemethod: factory function routinemethod(event), event is the HttpRequestEvent
        
        :param container: routine container. If None, default to self for bound method, or event.connection if not
         
        :param host: if specified, only response to request to specified host
        
        :param vhost: if specified, only response to request to specified vhost.
                      If not specified, response to dispatcher default vhost.
        
        :param method: if specified, response to specified methods
        '''
        regm = re.compile(path + b'$')
        if vhost is None:
            vhost = self.vhost
        if container is None:
            container = getattr(routinemethod, '__self__', None)
        def ismatch(event):
            # Check vhost
            if vhost is not None and getattr(event.createby, 'vhost', '') != vhost:
                return False
            # First parse the path
            # RFC said we should accept absolute path
            psplit = urlsplit(event.path)
            if psplit.path[:1] != b'/':
                # For security reason, ignore unrecognized path
                return False
            if psplit.netloc and host is not None and host != psplit.netloc:
                # Maybe a proxy request, ignore it
                return False
            if getattr(event.createby, 'unquoteplus', True):
                realpath = unquote_plus_to_bytes(psplit.path)
            else:
                realpath = unquote_to_bytes(psplit.path)
            m = regm.match(realpath)
            if m is None:
                return False
            event.realpath = realpath
            event.querystring = psplit.query
            event.path_match = m
            return True
        def func(event, scheduler):
            try:
                if event.canignore:
                    # Already processed
                    return
                event.canignore = True
                c = event.connection if container is None else container
                c.subroutine(routinemethod(event), False)
            except Exception:
                pass
        for m in method:
            self.registerHandler(HttpRequestEvent.createMatcher(host, None, m, _ismatch = ismatch), func)