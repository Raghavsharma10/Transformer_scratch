def getMiniHTML(self):
        '''
            getMiniHTML - Gets the HTML representation of this document without any pretty formatting
                and disregarding original whitespace beyond the functional.

                @return <str> - HTML with only functional whitespace present
        '''
        from .Formatter import AdvancedHTMLMiniFormatter
        html = self.getHTML()
        formatter = AdvancedHTMLMiniFormatter(None) # Do not double-encode
        formatter.feed(html)
        return formatter.getHTML()