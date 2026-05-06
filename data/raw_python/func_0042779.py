def url_tibiadata(self):
        """:class:`str`: The URL to the TibiaData.com page of the house."""
        return self.get_url_tibiadata(self.id, self.world) if self.id and self.world else None