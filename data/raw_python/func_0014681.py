def nextSibling(self):
        '''
            nextSibling - Returns the next sibling. This is the child following this node in the parent's list of children.

                    This could be text or an element. use nextSiblingElement to ensure element

                @return <None/str/AdvancedTag> - None if there are no nodes (text or tag) in the parent after this node,
                                                    Otherwise the following node (text or tag)
        '''
        parentNode = self.parentNode

        # If no parent, no siblings.
        if not parentNode:
            return None

        # Determine index in blocks
        myBlockIdx = parentNode.blocks.index(self)

        # If we are the last, no next sibling
        if myBlockIdx == len(parentNode.blocks) - 1:
            return None

        # Else, return the next block in parent
        return parentNode.blocks[ myBlockIdx + 1 ]