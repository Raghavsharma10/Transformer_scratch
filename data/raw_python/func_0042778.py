def url(self):
        """:class:`str`: The URL to the Tibia.com page of the house."""
        return self.get_url(self.id, self.world) if self.id and self.world else None