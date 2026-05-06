def multiply(self, number):
        """Return a Vector as the product of the vector and a real number."""
        return self.from_list([x * number for x in self.to_list()])