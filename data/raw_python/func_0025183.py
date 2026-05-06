def trajectory(self):
        """
        Calculates the center of mass for each time step and outputs an array

        Returns:

        """
        traj = np.zeros((2, self.times.size))
        for t, time in enumerate(self.times):
            traj[:, t] = self.center_of_mass(time)
        return traj