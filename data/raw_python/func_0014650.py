def createBlocksFromHTML(cls, html, encoding='utf-8'):
        '''
            createBlocksFromHTML - Returns the root level node (unless multiple nodes), and 
                a list of "blocks" added (text and nodes).

            @return list< str/AdvancedTag > - List of blocks created. May be strings (text nodes) or AdvancedTag (tags)

            NOTE:
                Results may be checked by:

                    issubclass(block.__class__, AdvancedTag)

                If True, block is a tag, otherwise, it is a text node
        '''
        
        parser = cls(encoding=encoding)

        parser.parseStr(html)

        rootNode = parser.getRoot()

        rootNode.remove()

        return rootNode.blocks