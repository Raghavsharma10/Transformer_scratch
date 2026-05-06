def handle_comment(self, comment):
        '''
            Internal for parsing
        '''
        inTag = self._inTag
        if len(inTag) > 0:
            inTag[-1].appendText('<!-- %s -->' %(comment,))
        else:
            raise MultipleRootNodeException()