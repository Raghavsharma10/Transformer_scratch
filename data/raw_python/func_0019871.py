def hasApplication(self, app):
        """Returns True if app is among the loaded modules.
        
        @param app: Module name.
        @return:    Boolean 
        
        """
        if self._applications is None:
            self._initApplicationList()
        return app in self._applications