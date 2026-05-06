def items(self):
        """D.items() -> list of D's (key, value) pairs, as 2-tuples"""
        return list(zip((self.scalars + self.sections), list(self.values())))