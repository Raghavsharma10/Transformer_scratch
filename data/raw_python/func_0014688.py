def textContent(self):
        '''
            textContent - property, gets the text of this node and all inner nodes.

                Use .innerText for just this node's text

              @return <str> - The text of all nodes at this level or lower
        '''

        def _collateText(curNode):
            '''
                _collateText - Recursive function to gather the "text" of all blocks

                                 in the order that they appear

                    @param curNode <AdvancedTag> - The current AdvancedTag to process

                    @return list<str> - A list of strings in order. Join using '' to obtain text
                                            as it would appear
            '''
                   
            curStrLst = []
            blocks = object.__getattribute__(curNode, 'blocks')

            for block in blocks:
                if isTagNode(block):
                    curStrLst += _collateText(block)
                else:
                    curStrLst.append(block)

            return curStrLst

        return ''.join(_collateText(self))