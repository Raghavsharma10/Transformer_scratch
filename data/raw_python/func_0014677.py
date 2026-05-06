def insertBefore(self, child, beforeChild):
        '''
            insertBefore - Inserts a child before #beforeChild


                @param child <AdvancedTag/str> - Child block to insert

                @param beforeChild <AdvancedTag/str> - Child block to insert before. if None, will  be appended

            @return - The added child. Note, if it is a text block (str), the return isl NOT be linked by reference.

            @raises ValueError - If #beforeChild is defined and is not a child of this node

        '''
        # When the second arg is null/None, the node is appended. The argument is required per JS API, but null is acceptable..
        if beforeChild is None:
            return self.appendBlock(child)

        # If #child is an AdvancedTag, we need to add it to both blocks and children.
        isChildTag = isTagNode(child)

        myBlocks = self.blocks
        myChildren = self.children

        # Find the index #beforeChild falls under current element
        try:
            blocksIdx =  myBlocks.index(beforeChild)
            if isChildTag:
                childrenIdx = myChildren.index(beforeChild)
        except ValueError:
            # #beforeChild is not a child of this element. Raise error.
            raise ValueError('Provided "beforeChild" is not a child of element, cannot insert.')
        
        # Add to blocks in the right spot
        self.blocks = myBlocks[:blocksIdx] + [child] + myBlocks[blocksIdx:]
        # Add to child in the right spot
        if isChildTag: 
            self.children = myChildren[:childrenIdx] + [child] + myChildren[childrenIdx:]
        
        return child