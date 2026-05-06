def matrix(self):
        """The 4x4 matrix representation of this rotation"""
        result = np.identity(4, float)
        result[0:3, 0:3] = self.r
        return result