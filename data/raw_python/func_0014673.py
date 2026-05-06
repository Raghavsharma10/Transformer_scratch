def appendInnerHTML(self, html):
        '''
            appendInnerHTML - Appends nodes from arbitrary HTML as if doing element.innerHTML += 'someHTML' in javascript.

            @param html <str> - Some HTML

            NOTE: If associated with a document ( AdvancedHTMLParser ), the html will use the encoding associated with
                    that document.

            @return - None. A browser would return innerHTML, but that's somewhat expensive on a high-level node.
              So just call .innerHTML explicitly if you need that
        '''

        # Late-binding to prevent circular import
        from .Parser import AdvancedHTMLParser

        # Inherit encoding from the associated document, if any.
        encoding = None
        if self.ownerDocument:
            encoding = self.ownerDocument.encoding

        # Generate blocks (text nodes and AdvancedTag's) from HTML
        blocks = AdvancedHTMLParser.createBlocksFromHTML(html, encoding)

        # Throw them onto this node
        self.appendBlocks(blocks)