def removeAttribute(self, attrName):
        '''
            removeAttribute - Removes an attribute, by name.
            
            @param attrName <str> - The attribute name

        '''
        attrName = attrName.lower()

        # Delete provided attribute name ( #attrName ) from attributes map
        try:
            del self._attributes[attrName]
        except KeyError:
            pass