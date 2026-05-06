def createElementFromHTML(cls, html, encoding='utf-8'):
        '''
            createElementFromHTML - Creates an element from a string of HTML.

                If this could create multiple root-level elements (children are okay),
                  you must use #createElementsFromHTML which returns a list of elements created.

            @param html <str> - Some html data

            @param encoding <str> - Encoding to use for document

            @raises MultipleRootNodeException - If given html would produce multiple root-level elements (use #createElementsFromHTML instead)

            @return AdvancedTag - A single AdvancedTag

            NOTE: If there is text outside the tag, they will be lost in this.
              Use createBlocksFromHTML instead if you need to retain both text and tags.

              Also, if you are just appending to an existing tag, use AdvancedTag.appendInnerHTML
        '''
        
        parser = cls(encoding=encoding)

        html = stripIEConditionals(html)
        try:
            HTMLParser.feed(parser, html)
        except MultipleRootNodeException:
            raise MultipleRootNodeException('Multiple nodes passed to createElementFromHTML method. Use #createElementsFromHTML instead to get a list of AdvancedTag elements.')

        rootNode = parser.getRoot()
        rootNode.remove()

        return rootNode