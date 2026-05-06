def removeTextAll(self, text):
        '''
            removeTextAll - Removes ALL occuraces of given text in a text node (i.e. not part of a tag)

            @param text <str> - text to remove

            @return list <str> - All text node containing #text BEFORE the text was removed.
                Empty list if no text removed

            NOTE: To remove a node, @see removeChild
            NOTE: To remove a block (maybe a node, maybe text), @see removeBlock
            NOTE: To remove a single occurace of text, @see removeText
        '''
        # TODO: This would be a good candidate for the refactor of text blocks
        removedBlocks = []

        blocks = self.blocks
        for i in range(len(blocks)):

            block = blocks[i]

            # We only care about text blocks
            if issubclass(block.__class__, AdvancedTag):
                continue

            if text in block:
                # Got a match, save a copy of the text block pre-replace for the return
                removedBlocks.append( block[:] )
                
                # And replace the text within this matched block
                blocks[i] = block.replace(text, '')

        
        # Regenerate self.text
        self.text = ''.join([thisBlock for thisBlock in blocks if not issubclass(thisBlock.__class__, AdvancedTag)])

        return removedBlocks