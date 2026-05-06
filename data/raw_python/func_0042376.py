def url(self):
        """:class:`str`: The URL to the highscores page on Tibia.com containing the results."""
        return self.get_url(self.world, self.category, self.vocation, self.page)