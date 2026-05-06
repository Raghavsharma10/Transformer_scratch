def getBlocksTags(self):
        '''
            getBlocksTags - Returns a list of tuples referencing the blocks which are direct children of this node, and the block is an AdvancedTag.

                The tuples are ( block, blockIdx ) where "blockIdx" is the index of self.blocks wherein the tag resides.

                @return list< tuple(block, blockIdx) > - A list of tuples of child blocks which are tags and their index in the self.blocks list
        '''
        myBlocks = self.blocks

        return [ (myBlocks[i], i) for i in range( len(myBlocks) ) if issubclass(myBlocks[i].__class__, AdvancedTag) ]