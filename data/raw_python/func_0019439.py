def evaluate(self, data):
        """ Estimate un-normalised probability density at target points
        
        Parameters
        ----------
        data : np.ndarray
            A `(num_targets, num_dim)` array of points to investigate. 
        
        Returns
        -------
        np.ndarray
            A `(num_targets)` length array of estimates

        Returns array of probability densities
        """
        if len(data.shape) == 1 and self.num_dim == 1:
            data = np.atleast_2d(data).T

        _d = np.dot(data - self.mean, self.A)

        # Get all points within range of kernels
        neighbors = self.tree.query_ball_point(_d, self.sigma * self.truncation)
        out = []
        for i, n in enumerate(neighbors):
            if len(n) >= self.nmin:
                diff = self.d[n, :] - _d[i]
                distsq = np.sum(diff * diff, axis=1)
            else:
                # If too few points get nmin closest
                dist, n = self.tree.query(_d[i], k=self.nmin)
                distsq = dist * dist
            out.append(np.sum(self.weights[n] * np.exp(self.sigma_fact * distsq)))
        return np.array(out)