def d(self, xi):
        """Convenience function to compute first derivative of basis functions. 'Memoized' for speed."""
        return self.__basis(xi, self.p, compute_derivatives=True)