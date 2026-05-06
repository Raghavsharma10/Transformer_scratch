def nextElementSibling(self):
        '''
            nextElementSibling - Returns the next sibling that is an element.
                This is the tag node following this node in the parent's list of children

                @return <None/AdvancedTag> - None if there are no children (tag) in the parent after this node,
                                                    Otherwise the following element (tag)
        '''
        parentNode = self.parentNode

        # If no parent, no siblings
        if not parentNode:
            return None

        # Determine the index in children
        myElementIdx = parentNode.children.index(self)

        # If we are last child, no next sibling
        if myElementIdx == len(parentNode.children) - 1:
            return None

        # Else, return the next child in parent
        return parentNode.children[myElementIdx+1]