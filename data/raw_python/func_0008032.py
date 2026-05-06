def __basis0(self, xi):
        """Order zero basis (for internal use)."""
        return np.where(np.all([self.knot_vector[:-1] <=  xi,
                                xi < self.knot_vector[1:]],axis=0), 1.0, 0.0)