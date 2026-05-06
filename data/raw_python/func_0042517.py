def by_player(self):
        """:class:`bool`: Whether the kill involves other characters."""
        return any([k.player and self.name != k.name for k in self.killers])