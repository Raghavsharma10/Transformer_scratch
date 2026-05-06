def lastChild(self):
        '''
            lastChild - property, Get the last child block, text or tag

                @return <str/AdvancedTag/None> - The last child block, or None if no child blocks
        '''
        blocks = object.__getattribute__(self, 'blocks')
        # First block is empty string for indent, but don't hardcode incase that changes
        if blocks[0] == '':
           firstIdx = 1
        else:
           firstIdx = 0

        if len(blocks) <= firstIdx:
            return None

        return blocks[-1]