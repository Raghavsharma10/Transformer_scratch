def handle_starttag(self, tagName, attributeList, isSelfClosing=False):
        '''
            Internal for parsing
        '''
        tagName = tagName.lower()
        inTag = self._inTag

        if isSelfClosing is False and tagName in IMPLICIT_SELF_CLOSING_TAGS:
            isSelfClosing = True

        newTag = AdvancedTag(tagName, attributeList, isSelfClosing, ownerDocument=self)
        if self.root is None:
            self.root = newTag
        elif len(inTag) > 0:
            inTag[-1].appendChild(newTag)
        else:
            raise MultipleRootNodeException()

        if isSelfClosing is False:
            inTag.append(newTag)

        return newTag