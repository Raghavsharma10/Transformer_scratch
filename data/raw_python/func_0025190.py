def estimate_motion(self, time, intensity_grid, max_u, max_v):
        """
        Estimate the motion of the object with cross-correlation on the intensity values from the previous time step.

        Args:
            time: time being evaluated.
            intensity_grid: 2D array of intensities used in cross correlation.
            max_u: Maximum x-component of motion. Used to limit search area.
            max_v: Maximum y-component of motion. Used to limit search area

        Returns:
            u, v, and the minimum error.
        """
        ti = np.where(time == self.times)[0][0]
        mask_vals = np.where(self.masks[ti].ravel() == 1)
        i_vals = self.i[ti].ravel()[mask_vals]
        j_vals = self.j[ti].ravel()[mask_vals]
        obj_vals = self.timesteps[ti].ravel()[mask_vals]
        u_shifts = np.arange(-max_u, max_u + 1)
        v_shifts = np.arange(-max_v, max_v + 1)
        min_error = 99999999999.0
        best_u = 0
        best_v = 0
        for u in u_shifts:
            j_shift = j_vals - u
            for v in v_shifts:
                i_shift = i_vals - v
                if np.all((0 <= i_shift) & (i_shift < intensity_grid.shape[0]) &
                                  (0 <= j_shift) & (j_shift < intensity_grid.shape[1])):
                    shift_vals = intensity_grid[i_shift, j_shift]
                else:
                    shift_vals = np.zeros(i_shift.shape)
                # This isn't correlation; it is mean absolute error.
                error = np.abs(shift_vals - obj_vals).mean()
                if error < min_error:
                    min_error = error
                    best_u = u * self.dx
                    best_v = v * self.dx
        # 60 seems arbitrarily high
        #if min_error > 60:
        #    best_u = 0
        #    best_v = 0
        self.u[ti] = best_u
        self.v[ti] = best_v
        return best_u, best_v, min_error