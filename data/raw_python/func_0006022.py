def visit(self, url=''):
        """Visit the url, checking for rr errors in the response

        @param url: URL
        @return: Visit result
        """
        result = super(CoyoteDriver, self).visit(url)
        source = self.page_source()
        return result