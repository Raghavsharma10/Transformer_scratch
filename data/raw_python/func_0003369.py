def registerAPIs(self, apidefs):
        '''
        API definition is in format: `(name, handler, container, discoverinfo)`
        
        if the handler is a generator, container should be specified
        handler should accept two arguments::
        
            def handler(name, params):
                ...
        
        `name` is the method name, `params` is a dictionary contains the parameters.
        
        the handler can either return the result directly, or be a generator (async-api),
        and write the result to container.retvalue on exit.
        e.g::
        
            ('method1', self.method1),    # method1 directly returns the result
            ('method2', self.method2, self) # method2 is an async-api
        
        Use api() to automatically generate API definitions.
        '''
        handlers = [self._createHandler(*apidef) for apidef in apidefs]
        self.handler.registerAllHandlers(handlers)
        self.discoverinfo.update((apidef[0], apidef[3] if len(apidef) > 3 else {'description':cleandoc(apidef[1].__doc__)}) for apidef in apidefs)