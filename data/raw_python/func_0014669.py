def removeBlocks(self, blocks):
        '''
            removeBlock - Removes a list of blocks (the first occurance of each) from the direct children of this node.

            @param blocks  list<str/AdvancedTag> - List of AdvancedTags for tag nodes, else strings for text nodes

            @return The removed blocks in each slot, or None if None removed.

            @see removeChild
            @see removeText

            For multiple, @see removeBlocks
        '''
        ret = []
        for block in blocks:
            if issubclass(block.__class__, AdvancedTag):
                ret.append( self.removeChild(block) )
            else:
                # TODO: Should this just forward to removeText?
                ret.append( self.removeBlock(block) )

        return ret