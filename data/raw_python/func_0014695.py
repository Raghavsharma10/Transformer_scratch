def getEndTag(self):
        '''
            getEndTag - returns the end tag representation as HTML string

            @return - String of end tag
        '''
        # If this is a self-closing tag, we have no end tag (opens and closes in the start)
        if self.isSelfClosing is True:
            return ''

        tagName = self.tagName

        # Do not add any indentation to the end of preformatted tags.
        if self._indent and tagName in PREFORMATTED_TAGS:
            return "</%s>" %(tagName, )

        # Otherwise, indent the end of this tag
        return "%s</%s>" %(self._indent, tagName)