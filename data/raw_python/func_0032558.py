def getJSModuleURL(self, moduleName):
        """
        Retrieve an L{URL} object which references the given module name.

        This makes a 'best effort' guess as to an fully qualified HTTPS URL
        based on the hostname provided during rendering and the configuration
        of the site.  This is to avoid unnecessary duplicate retrieval of the
        same scripts from two different URLs by the browser.

        If such configuration does not exist, however, it will simply return an
        absolute path URL with no hostname or port.

        @raise NotImplementedError: if rendering has not begun yet and
        therefore beforeRender has not provided us with a usable hostname.
        """
        if self._moduleRoot is None:
            raise NotImplementedError(
                "JS module URLs cannot be requested before rendering.")
        moduleHash = self.hashCache.getModule(moduleName).hashValue
        return self._moduleRoot.child(moduleHash).child(moduleName)