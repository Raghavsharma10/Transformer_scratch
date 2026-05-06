def parseStr(self, html):
        '''
            parseStr - Parses a string and creates the DOM tree and indexes.

                @param html <str> - valid HTML
        '''
        self.reset()

        if isinstance(html, bytes):
            self.feed(html.decode(self.encoding))
        else:
            self.feed(html)