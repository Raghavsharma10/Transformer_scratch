def _checkAttribs(self, strict):
        """Check initial attributes to make sure they are legal"""
        if self.choice:
            warning("Choice values not allowed for real-type parameter " +
                    self.name, strict)
            self.choice = None