def innerHTML(self):
        '''
            innerHTML - Returns an HTML string of the inner contents of this tag, including children.

            @return - String of inner contents HTML
        '''

        # If a self-closing tag, there are no contents
        if self.isSelfClosing is True:
            return ''

        # Assemble all the blocks.
        ret = []

        # Iterate through blocks
        for block in self.blocks:
            # For each block:
            #   If a tag, append the outer html (start tag, contents, and end tag)
            #   Else, append the text node directly

            if isinstance(block, AdvancedTag):
                ret.append(block.outerHTML)
            else:
                ret.append(block)
        
        return ''.join(ret)