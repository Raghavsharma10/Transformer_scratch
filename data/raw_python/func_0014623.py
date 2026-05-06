def handle_endtag(self, tagName):
        '''
            Internal for parsing
        '''
        try:
            foundIt = False
            inTag = self._inTag
            for i in range(len(inTag)):
                if inTag[i].tagName == tagName:
                    foundIt = True
                    break

            if not foundIt:
                return
            # Handle closing tags which should have been closed but weren't
            while inTag[-1].tagName != tagName:
                inTag.pop()

            inTag.pop()
        except:
            pass