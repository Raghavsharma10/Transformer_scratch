def appendBlock(self, block):
        '''
            append / appendBlock - Append a block to this element. A block can be a string (text node), or an AdvancedTag (tag node)

            @param <str/AdvancedTag> - block to add

            @return - #block

            NOTE: To add multiple blocks, @see appendBlocks
                  If you know the type, use either @see appendChild for tags or @see appendText for text
        '''
        # Determine block type and call appropriate method
        if isinstance(block, AdvancedTag):
            self.appendNode(block)
        else:
            self.appendText(block)
        
        return block