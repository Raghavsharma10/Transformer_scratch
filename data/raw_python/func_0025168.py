def period_neighborhood_probability(self, radius, smoothing, threshold, stride,start_time,end_time):
        """
        Calculate the neighborhood probability over the full period of the forecast

        Args:
            radius: circular radius from each point in km
            smoothing: width of Gaussian smoother in km
            threshold: intensity of exceedance
            stride: number of grid points to skip for reduced neighborhood grid

        Returns:
            (neighborhood probabilities)
        """
        neighbor_x = self.x[::stride, ::stride]
        neighbor_y = self.y[::stride, ::stride]
        neighbor_kd_tree = cKDTree(np.vstack((neighbor_x.ravel(), neighbor_y.ravel())).T)
        neighbor_prob = np.zeros((self.data.shape[0], neighbor_x.shape[0], neighbor_x.shape[1]))
        print('Forecast Hours: {0}-{1}'.format(start_time, end_time))
        for m in range(len(self.members)):
            period_max = self.data[m,start_time:end_time,:,:].max(axis=0)
            valid_i, valid_j = np.where(period_max >= threshold)
            print(self.members[m], len(valid_i))
            if len(valid_i) > 0:
                var_kd_tree = cKDTree(np.vstack((self.x[valid_i, valid_j], self.y[valid_i, valid_j])).T)
                exceed_points = np.unique(np.concatenate(var_kd_tree.query_ball_tree(neighbor_kd_tree, radius))).astype(int)
                exceed_i, exceed_j = np.unravel_index(exceed_points, neighbor_x.shape)
                neighbor_prob[m][exceed_i, exceed_j] = 1
                if smoothing > 0:
                    neighbor_prob[m] = gaussian_filter(neighbor_prob[m], smoothing,mode='constant')
        return neighbor_prob