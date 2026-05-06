def getHTML(self):
        '''
            getHTML - Get the full HTML as contained within this tree.

                If parsed from a document, this will contain the original whitespacing.

                @returns - <str> of html

                    @see getFormattedHTML

                    @see getMiniHTML
        '''
        root = self.getRoot()
        if root is None:
            raise ValueError('Did not parse anything. Use parseFile or parseStr')

        if self.doctype:
            doctypeStr = '<!%s>\n' %(self.doctype)
        else:
            doctypeStr = ''

        # 6.6.0: If we have a real root tag, print the outerHTML. If we have a fake root tag (for multiple root condition),
        #   then print the innerHTML (skipping the outer root tag). Otherwise, we will miss
        #   untagged text (between the multiple root nodes).
        rootNode = self.getRoot()
        if rootNode.tagName == INVISIBLE_ROOT_TAG:
            return doctypeStr + rootNode.innerHTML
        else:
            return doctypeStr + rootNode.outerHTML