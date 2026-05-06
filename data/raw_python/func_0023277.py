def normalized(self):
        """Return a Rect covering the same area, but with height and width
        guaranteed to be positive."""
        return Rect(pos=(min(self.left, self.right),
                         min(self.top, self.bottom)),
                    size=(abs(self.width), abs(self.height)))