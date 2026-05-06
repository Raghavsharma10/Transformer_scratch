def getConfig(self):
        """Returns dictionary of config entries for Munin Graph.
        
        @return: Dictionary of config entries. 
        
        """
        return {'graph': self._graphAttrDict,
                'fields': [(field_name, self._fieldAttrDict.get(field_name))
                           for field_name in self._fieldNameList]}