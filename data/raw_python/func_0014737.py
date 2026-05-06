def handle_endtag(self, tagName):
        '''
            handle_endtag - Internal for parsing
        '''
        inTag = self._inTag
        try:
            # Handle closing tags which should have been closed but weren't
            foundIt = False
            for i in range(len(inTag)):
                if inTag[i].tagName == tagName:
                    foundIt = True
                    break

            if not foundIt:
                sys.stderr.write('WARNING: found close tag with no matching start.\n')
                return

            while inTag[-1].tagName != tagName:
                oldTag = inTag.pop()
                if oldTag.tagName in PREFORMATTED_TAGS:
                    self.inPreformatted -= 1

                self.currentIndentLevel -= 1

            inTag.pop()
            if tagName != INVISIBLE_ROOT_TAG:
                self.currentIndentLevel -= 1
            if tagName in PREFORMATTED_TAGS:
                self.inPreformatted -= 1
        except:
            pass