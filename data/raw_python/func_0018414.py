def choose(self):
        """Marks the item as the one the user is in."""
        if not self.choosed:
            self.choosed = True
            self.pos = self.pos + Sep(5, 0)