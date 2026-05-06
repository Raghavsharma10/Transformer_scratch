def _setup_grid(self, cutoff, unit_cell, grid):
        """Choose a proper grid for the binning process"""
        if grid is None:
            # automatically choose a decent grid
            if unit_cell is None:
                grid = cutoff/2.9
            else:
                # The following would be faster, but it is not reliable
                # enough yet.
                #grid = unit_cell.get_optimal_subcell(cutoff/2.0)
                divisions = np.ceil(unit_cell.spacings/cutoff)
                divisions[divisions<1] = 1
                grid = unit_cell/divisions

        if isinstance(grid, float):
            grid_cell = UnitCell(np.array([
                [grid, 0, 0],
                [0, grid, 0],
                [0, 0, grid]
            ]))
        elif isinstance(grid, UnitCell):
            grid_cell = grid
        else:
            raise TypeError("Grid must be None, a float or a UnitCell instance.")

        if unit_cell is not None:
            # The columns of integer_matrix are the unit cell vectors in
            # fractional coordinates of the grid cell.
            integer_matrix = grid_cell.to_fractional(unit_cell.matrix.transpose()).transpose()
            if abs((integer_matrix - np.round(integer_matrix))*self.unit_cell.active).max() > 1e-6:
                raise ValueError("The unit cell vectors are not an integer linear combination of grid cell vectors.")
            integer_matrix = integer_matrix.round()
            integer_cell = UnitCell(integer_matrix, unit_cell.active)
        else:
            integer_cell = None

        return grid_cell, integer_cell