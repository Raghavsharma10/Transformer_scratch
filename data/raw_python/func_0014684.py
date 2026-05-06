def previousElementSibling(self):
        '''
            previousElementSibling - Returns the previous  sibling  that is an element. 

                                        This is the previous tag node in the parent's list of children


                @return <None/AdvancedTag> - None if there are no children (tag) in the parent before this node,
                                                    Otherwise the previous element (tag)

        '''
        parentNode = self.parentNode

        # If no parent, no siblings
        if not parentNode:
            return None

        # Determine this node's index in the children of parent
        myElementIdx = parentNode.children.index(self)
        
        # If we are the first child, no previous element
        if myElementIdx == 0:
            return None

        # Else, return previous element tag
        return parentNode.children[myElementIdx-1]