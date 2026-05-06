def handle_starttag(self, tagName, attributeList, isSelfClosing=False):
        '''
            internal for parsing
        '''
        newTag = AdvancedHTMLParser.handle_starttag(self, tagName, attributeList, isSelfClosing)
        self._indexTag(newTag)

        return newTag