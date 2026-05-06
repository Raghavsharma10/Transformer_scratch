def handle_data(self, data):
        '''
            Internal for parsing
        '''
        if data:
            inTag = self._inTag
            if len(inTag) > 0:
                inTag[-1].appendText(data)
            elif data.strip(): #and not self.getRoot():
                # Must be text prior to or after root node
                raise MultipleRootNodeException()