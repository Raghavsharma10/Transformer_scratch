def createElementsFromHTML(cls, html, encoding='utf-8'):
        '''
            createElementsFromHTML - Creates elements from provided html, and returns a list of the root-level elements
                children of these root-level nodes are accessable via the usual means.

            @param html <str> - Some html data

            @param encoding <str> - Encoding to use for document

            @return list<AdvancedTag> - The root (top-level) tags from parsed html.

            NOTE: If there is text outside the tags, they will be lost in this.
              Use createBlocksFromHTML instead if you need to retain both text and tags.

              Also, if you are just appending to an existing tag, use AdvancedTag.appendInnerHTML
        '''
        # TODO: If text is present outside a tag, it will be lost.

        parser = cls(encoding=encoding)

        parser.parseStr(html)

        rootNode = parser.getRoot()

        rootNode.remove() # Detatch from temp document

        if isInvisibleRootTag(rootNode):
            return rootNode.children

        return [rootNode]