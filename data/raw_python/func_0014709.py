def getElementsByClassName(self, className):
        '''
            getElementsByClassName - Search children of this tag for tags containing a given class name

            @param className - Class name

            @return - TagCollection of matching elements
        '''
        elements = []
        for child in self.children:
            if child.hasClass(className) is True:
                elements.append(child)
            elements += child.getElementsByClassName(className)
        return TagCollection(elements)