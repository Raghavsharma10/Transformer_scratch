def handle_entityref(self, entity):
        '''
            Internal for parsing
        '''
        inTag = self._inTag
        if len(inTag) > 0:
            inTag[-1].appendText('&%s;' %(entity,))
        else:
            raise MultipleRootNodeException()