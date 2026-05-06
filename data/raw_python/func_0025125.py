def dilate_obs(self, dilation_radius):
        """
        Use a dilation filter to grow positive observation areas by a specified number of grid points

        :param dilation_radius: Number of times to dilate the grid.
        :return:
        """
        for s in self.size_thresholds:
            self.dilated_obs[s] = np.zeros(self.window_obs[self.mrms_variable].shape)
            for t in range(self.dilated_obs[s].shape[0]):
                self.dilated_obs[s][t][binary_dilation(self.window_obs[self.mrms_variable][t] >= s, iterations=dilation_radius)] = 1