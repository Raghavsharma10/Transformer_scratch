def hasChannelType(self, chan):
        """Returns True if chan is among the supported channel types.
        
        @param app: Module name.
        @return:    Boolean 
        
        """
        if self._chantypes is None:
            self._initChannelTypesList()
        return chan in self._chantypes