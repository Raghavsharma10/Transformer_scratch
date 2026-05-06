def removeBlock(self, block):
        '''
            removeBlock - Removes a single block (text node or AdvancedTag) which is a child of this object.

            @param block <str/AdvancedTag> - The block (text node or AdvancedTag) to remove.
            
            @return Returns the removed block if one was removed, or None if requested block is not a child of this node.

            NOTE: If you know you are going to remove an AdvancedTag, @see removeChild
                  If you know you are going to remove a text node,    @see removeText

            If removing multiple blocks, @see removeBlocks
        '''
        if issubclass(block.__class__, AdvancedTag):
            return self.removeChild(block)
        else:
            return self.removeText(block)