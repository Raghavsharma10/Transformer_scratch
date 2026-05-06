def registerAPI(self, name, handler, container = None, discoverinfo = None, criteria = None):
        """
        Append new API to this handler
        """
        self.handler.registerHandler(*self._createHandler(name, handler, container, criteria))
        if discoverinfo is None:
            self.discoverinfo[name] = {'description': cleandoc(handler.__doc__)}
        else:
            self.discoverinfo[name] = discoverinfo