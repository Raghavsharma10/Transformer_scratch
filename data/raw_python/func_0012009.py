def _links(self):
        """Calculate total energy production. Not Rounded"""
        total = 0.0
        for value in self.link.values():
            total += value
        return total