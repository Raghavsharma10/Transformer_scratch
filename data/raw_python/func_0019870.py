def hasModule(self, mod):
        """Returns True if mod is among the loaded modules.
        
        @param mod: Module name.
        @return:    Boolean 
        
        """
        if self._modules is None:
            self._initModuleList()
        return mod in self._modules