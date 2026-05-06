def tagBlocks(self):
        '''
            tagBlocks - Property. 
                        Returns all the blocks which are direct children of this node, where that block is a tag (not text)

                NOTE: This is similar to .children , and you should probably use .children instead except within this class itself

                @return list<AdvancedTag> - A list of direct children which are tags.
        '''
        myBlocks = self.blocks

        return [block for block in myBlocks if issubclass(block.__class__, AdvancedTag)]