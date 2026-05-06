def get_hessian(self):
        """Return the hessian"""
        force_const = self.fields.get("Cartesian Force Constants")
        if force_const is None:
            return None
        N = len(self.molecule.numbers)
        result = np.zeros((3*N, 3*N), float)
        counter = 0
        for row in range(3*N):
            result[row, :row+1] = force_const[counter:counter+row+1]
            result[:row+1, row] = force_const[counter:counter+row+1]
            counter += row + 1
        return result