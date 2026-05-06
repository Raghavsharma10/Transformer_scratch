def getStartTag(self):
        '''
            getStartTag - Returns the start tag represented as HTML

            @return - String of start tag with attributes
        '''
        attributeStrings = []
        # Get all attributes as a tuple (name<str>, value<str>)
        for name, val in self._attributes.items():
            # Get all attributes
            if val:
                val = tostr(val)

            # Only binary attributes have a "present/not present"
            if val or name not in TAG_ITEM_BINARY_ATTRIBUTES:
                # Escape any quotes found in the value
                val = escapeQuotes(val)

                # Add a name="value" to the resulting string
                attributeStrings.append('%s="%s"' %(name, val) )
            else:
                # This is a binary attribute, and thus only includes the name ( e.x. checked )
                attributeStrings.append(name)

        # Join together all the attributes in @attributeStrings list into a string
        if attributeStrings:
            attributeString = ' ' + ' '.join(attributeStrings)
        else:
            attributeString = ''

        # If this is a self-closing tag, generate like  <tag attr1="val" attr2="val2" />  with the close "/>"
        # Include the indent prior to tag opening
        if self.isSelfClosing is False:
            return "%s<%s%s >" %(self._indent, self.tagName, attributeString)
        else:
            return "%s<%s%s />" %(self._indent, self.tagName, attributeString)