def stop_choose(self):
        """Marks the item as the one the user is not in."""
        if self.choosed:
            self.choosed = False
            self.pos = self.pos + Sep(-5, 0)