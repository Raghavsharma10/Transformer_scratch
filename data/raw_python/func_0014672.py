def appendBlocks(self, blocks):
        '''
            appendBlocks - Append blocks to this element. A block can be a string (text node), or an AdvancedTag (tag node)

            @param blocks list<str/AdvancedTag> - A list, in order to append, of blocks to add.

            @return - #blocks

            NOTE: To add a single block, @see appendBlock
                  If you know the type, use either @see appendChild for tags or @see appendText for text
        '''
        for block in blocks:
            if isinstance(block, AdvancedTag):
                self.appendNode(block)
            else:
                self.appendText(block)

        return blocks