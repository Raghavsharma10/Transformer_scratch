def getAttributesList(self):
        '''
            getAttributesList - Get a copy of all attributes as a list of tuples (name, value)

              ALL values are converted to string and copied, so modifications will not affect the original attributes.
                If you want types like "style" to work as before, you'll need to recreate those elements (like StyleAttribute(strValue) ).

              @return list< tuple< str(name), str(value) > > - A list of tuples of attrName, attrValue pairs, all converted to strings.

                This is suitable for passing back into AdvancedTag when creating a new tag.
        '''
        return [ (tostr(name)[:], tostr(value)[:]) for name, value in self._attributes.items() ]