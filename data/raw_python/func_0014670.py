def appendChild(self, child):
        '''
            appendChild - Append a child to this element.

            @param child <AdvancedTag> - Append a child element to this element
        '''

        # Associate parentNode of #child to this tag
        child.parentNode = self

        # Associate owner document to child and all children recursive
        ownerDocument = self.ownerDocument

        child.ownerDocument = ownerDocument
        for subChild in child.getAllChildNodes():
            subChild.ownerDocument = ownerDocument

        # Our tag cannot be self-closing if we have a child tag
        self.isSelfClosing = False

        # Append to both "children" and "blocks"
        self.children.append(child)
        self.blocks.append(child)
        return child