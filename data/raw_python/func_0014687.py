def textBlocks(self):
        '''
            textBlocks - Property. 
                        Returns all the blocks which are direct children of this node, where that block is a text (not a tag)

                @return list<AdvancedTag> - A list of direct children which are text.
        '''
        myBlocks = self.blocks

        return [block for block in myBlocks if not issubclass(block.__class__, AdvancedTag)]