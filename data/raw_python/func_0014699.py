def getAttributesDict(self):
        '''
            getAttributesDict - Get a copy of all attributes as a dict map of name -> value

              ALL values are converted to string and copied, so modifications will not affect the original attributes.
                If you want types like "style" to work as before, you'll need to recreate those elements (like StyleAttribute(strValue) ).

              @return <dict ( str(name), str(value) )> - A dict of attrName to attrValue , all as strings and copies.
        '''
            
        return { tostr(name)[:] : tostr(value)[:] for name, value in self._attributes.items() }