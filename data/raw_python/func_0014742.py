def handle_endtag(self, tagName):
        '''
            Internal for parsing
        '''
        inTag = self._inTag
        if len(inTag) == 0:
            # Attempted to close, but no open tags
            raise InvalidCloseException(tagName, [])

        foundIt = False
        i = len(inTag) - 1
        while i >= 0:
            if inTag[i].tagName == tagName:
                foundIt = True
                break
            i -= 1

        if not foundIt:
            # Attempted to close, but did not match anything
            raise InvalidCloseException(tagName, inTag)

        if inTag[-1].tagName != tagName:
            raise MissedCloseException(tagName, [x for x in inTag[-1 * (i+1): ] ] )

        inTag.pop()