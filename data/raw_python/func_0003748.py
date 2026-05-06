def add_cell_vector(self, vector):
        """Returns a new unit cell with an additional cell vector"""
        act = self.active_inactive[0]
        if len(act) == 3:
            raise ValueError("The unit cell already has three active cell vectors.")
        matrix = np.zeros((3, 3), float)
        active = np.zeros(3, bool)
        if len(act) == 0:
            # Add the new vector
            matrix[:, 0] = vector
            active[0] = True
            return UnitCell(matrix, active)

        a = self.matrix[:, act[0]]
        matrix[:, 0] = a
        active[0] = True
        if len(act) == 1:
            # Add the new vector
            matrix[:, 1] = vector
            active[1] = True
            return UnitCell(matrix, active)

        b = self.matrix[:, act[1]]
        matrix[:, 1] = b
        active[1] = True
        if len(act) == 2:
            # Add the new vector
            matrix[:, 2] = vector
            active[2] = True
            return UnitCell(matrix, active)