def getElementsWithAttrValues(self, attrName, attrValues):
        '''
            getElementsWithAttrValues - Search children of this tag for tags with an attribute name and one of several values

            @param attrName <lowercase str> - Attribute name (lowercase)
            @param attrValues set<str> - set of acceptable attribute values

            @return - TagCollection of matching elements
        '''
        elements = []

        for child in self.children:
            if child.getAttribute(attrName) in attrValues:
                elements.append(child)
            elements += child.getElementsWithAttrValues(attrName, attrValues)
        return TagCollection(elements)