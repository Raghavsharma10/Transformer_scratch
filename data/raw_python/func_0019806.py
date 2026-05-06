def hasField(self, name):
        """Returns true if field with field_name exists.
        
        @param name: Field Name
        @return:     Boolean
        
        """
        if self._autoFixNames:
            name = self._fixName(name)
        return self._fieldAttrDict.has_key(name)