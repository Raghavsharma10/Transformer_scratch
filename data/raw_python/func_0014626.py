def handle_charref(self, charRef):
        '''
            Internal for parsing
        '''
        inTag = self._inTag
        if len(inTag) > 0:
            inTag[-1].appendText('&#%s;' %(charRef,))
        else:
            raise MultipleRootNodeException()