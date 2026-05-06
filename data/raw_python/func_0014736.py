def handle_starttag(self, tagName, attributeList, isSelfClosing=False):
        '''
            handle_starttag - Internal for parsing
        '''
        tagName = tagName.lower()
        inTag = self._inTag

        if isSelfClosing is False and tagName in IMPLICIT_SELF_CLOSING_TAGS:
            isSelfClosing = True

        newTag = AdvancedTag(tagName, attributeList, isSelfClosing)
        if self.root is None:
            self.root = newTag
        elif len(inTag) > 0:
            inTag[-1].appendChild(newTag)
        else:
            raise MultipleRootNodeException()

        if self.inPreformatted is 0:
            newTag._indent = self._getIndent()

        if tagName in PREFORMATTED_TAGS:
            self.inPreformatted += 1

        if isSelfClosing is False:
            inTag.append(newTag)
            if tagName != INVISIBLE_ROOT_TAG:
                self.currentIndentLevel += 1