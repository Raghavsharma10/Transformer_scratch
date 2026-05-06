def get_sisters(self):
        """ Returns an indepent list of sister nodes."""
        if self.up != None:
            return [ch for ch in self.up.children if ch != self]
        else:
            return []