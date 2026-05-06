def previousSibling(self):
        '''
            previousSibling - Returns the previous sibling. This would be the previous node (text or tag) in the parent's list
            
                This could be text or an element. use previousSiblingElement to ensure element


                @return <None/str/AdvancedTag> - None if there are no nodes (text or tag) in the parent before this node,
                                                    Otherwise the previous node (text or tag)
        '''
        parentNode = self.parentNode

        # If no parent, no previous sibling
        if not parentNode:
            return None

        # Determine block index on parent of this node
        myBlockIdx = parentNode.blocks.index(self)
        
        # If we are the first, no previous sibling
        if myBlockIdx == 0:
            return None

        # Else, return the previous block in parent
        return parentNode.blocks[myBlockIdx-1]