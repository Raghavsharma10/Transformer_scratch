def removeText(self, text):
        '''
            removeText - Removes the first occurace of given text in a text node (i.e. not part of a tag)

            @param text <str> - text to remove

            @return text <str/None> - The text in that block (text node) after remove, or None if not found

            NOTE: To remove a node, @see removeChild
            NOTE: To remove a block (maybe a node, maybe text), @see removeBlock
            NOTE: To remove ALL occuraces of text, @see removeTextAll
        '''
        # TODO: This would be a good candidate for the refactor of text blocks
        removedBlock = None

        # Scan all text blocks for "text"
        blocks = self.blocks
        for i in range(len(blocks)):
            block = blocks[i]

            # We only care about text blocks
            if issubclass(block.__class__, AdvancedTag):
                continue

            if text in block:
                # We have a block that matches.
                
                # Create a copy of the old text in this block for return
                removedBlock = block[:]
                # Remove first occurance of #text from matched block
                blocks[i] = block.replace(text, '')
                break # remove should only remove FIRST occurace, per other methods

        # Regenerate the "text" property
        self.text = ''.join([thisBlock for thisBlock in blocks if not issubclass(thisBlock.__class__, AdvancedTag)])

        # Return None if no match, otherwise the text previously within the block we removed #text from
        return removedBlock