def getAttribute(self, attrName, defaultValue=None):
        '''
            getAttribute - Gets an attribute on this tag. Be wary using this for classname, maybe use addClass/removeClass. Attribute names are all lowercase.
                @return - The attribute value, or None if none exists.
        '''

        if attrName in TAG_ITEM_BINARY_ATTRIBUTES:
            if attrName in self._attributes:
                attrVal = self._attributes[attrName]
                if not attrVal:
                    return True # Empty valued binary attribute

                return attrVal # optionally-valued binary attribute
            else:
                return False
        else:
            return self._attributes.get(attrName, defaultValue)