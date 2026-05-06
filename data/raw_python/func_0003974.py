def inverse(self):
        """Returns the inverse bijection."""
        result = self.__class__()
        result.forward = copy.copy(self.reverse)
        result.reverse = copy.copy(self.forward)
        return result