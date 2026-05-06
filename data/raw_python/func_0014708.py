def getElementsByAttr(self, attrName, attrValue):
        '''
            getElementsByAttr - Search children of this tag for tags with an attribute name/value pair

            @param attrName - Attribute name (lowercase)
            @param attrValue - Attribute value

            @return - TagCollection of matching elements
        '''
        elements = []
        for child in self.children:
            if child.getAttribute(attrName) == attrValue:
                elements.append(child)
            elements += child.getElementsByAttr(attrName, attrValue)
        return TagCollection(elements)