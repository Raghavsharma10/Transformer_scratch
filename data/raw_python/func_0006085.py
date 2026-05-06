def build(self, **variables):
        """Formats the locator with specified parameters"""
        return Locator(self.by, self.locator.format(**variables), self.description)