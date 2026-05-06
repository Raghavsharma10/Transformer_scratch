def insertAfter(self, child, afterChild):
        '''
            insertAfter - Inserts a child after #afterChild


                @param child <AdvancedTag/str> - Child block to insert

                @param afterChild <AdvancedTag/str> - Child block to insert after. if None, will  be appended

            @return - The added child. Note, if it is a text block (str), the return isl NOT be linked by reference.
        '''

        # If after child is null/None, just append
        if afterChild is None:
            return self.appendBlock(child)

        isChildTag = isTagNode(child)

        myBlocks = self.blocks
        myChildren = self.children

        # Determine where we need to insert this both in "blocks" and, if a tag, "children"
        try:
            blocksIdx =  myBlocks.index(afterChild)
            if isChildTag:
                childrenIdx = myChildren.index(afterChild)
        except ValueError:
            raise ValueError('Provided "afterChild" is not a child of element, cannot insert.')

        # Append child to requested spot
        self.blocks = myBlocks[:blocksIdx+1] + [child] + myBlocks[blocksIdx+1:]
        if isChildTag:
            self.children = myChildren[:childrenIdx+1] + [child] + myChildren[childrenIdx+1:]

        return child