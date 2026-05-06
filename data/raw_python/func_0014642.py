def getFormattedHTML(self, indent='  '):
        '''
            getFormattedHTML - Get formatted and xhtml of this document, replacing the original whitespace
                with a pretty-printed version

            @param indent - space/tab/newline of each level of indent, or integer for how many spaces per level
        
            @return - <str> Formatted html

            @see getHTML - Get HTML with original whitespace

            @see getMiniHTML - Get HTML with only functional whitespace remaining
        '''
        from .Formatter import AdvancedHTMLFormatter
        html = self.getHTML()
        formatter = AdvancedHTMLFormatter(indent, None) # Do not double-encode
        formatter.feed(html)
        return formatter.getHTML()