def category(self, category):
        """
        Change category of current search and return self
        """
        self.url.category = category
        self.url.set_page(1)
        return self