def hasAttribute(self, attrName):
        '''
            hasAttribute - Checks for the existance of an attribute. Attribute names are all lowercase.
   
                @param attrName <str> - The attribute name
                
                @return <bool> - True or False if attribute exists by that name
        '''
        attrName = attrName.lower()

        # Check if requested attribute is present on this node
        return bool(attrName in self._attributes)