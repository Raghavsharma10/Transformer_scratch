def all(self):
        """
        Yield torrents in range from current page to last page
        """
        return self.pages(self.url.page, self.url.max_page)