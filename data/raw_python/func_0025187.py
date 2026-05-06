def max_intensity(self, time):
        """
        Calculate the maximum intensity found at a timestep.

        """
        ti = np.where(time == self.times)[0][0]
        return self.timesteps[ti].max()