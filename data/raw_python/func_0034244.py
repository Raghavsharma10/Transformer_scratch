def same_player(self, other):
        """
        Compares name and color.
        Returns True if both are owned by the same player.
        """
        return self.name == other.name \
            and self.color == other.color