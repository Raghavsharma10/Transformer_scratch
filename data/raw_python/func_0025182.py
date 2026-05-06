def center_of_mass(self, time):
        """
        Calculate the center of mass at a given timestep.

        Args:
            time: Time at which the center of mass calculation is performed

        Returns:
            The x- and y-coordinates of the center of mass.
        """
        if self.start_time <= time <= self.end_time:
            diff = time - self.start_time
            valid = np.flatnonzero(self.masks[diff] != 0)
            if valid.size > 0:
                com_x = 1.0 / self.timesteps[diff].ravel()[valid].sum() * np.sum(self.timesteps[diff].ravel()[valid] *
                                                                                 self.x[diff].ravel()[valid])
                com_y = 1.0 / self.timesteps[diff].ravel()[valid].sum() * np.sum(self.timesteps[diff].ravel()[valid] *
                                                                                 self.y[diff].ravel()[valid])
            else:
                com_x = np.mean(self.x[diff])
                com_y = np.mean(self.y[diff])
        else:
            com_x = None
            com_y = None
        return com_x, com_y