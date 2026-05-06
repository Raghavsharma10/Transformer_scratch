def url_tibiadata(self):
        """:class:`str`: The URL to the highscores page on TibiaData.com containing the results."""
        return self.get_url_tibiadata(self.world, self.category, self.vocation)