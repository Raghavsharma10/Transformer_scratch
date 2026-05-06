def autopage(self):
        """Iterate through results from all pages.

        :return: all results
        :rtype: generator
        """
        while self.items:
            yield from self.items
            self.items = self.fetch_next()