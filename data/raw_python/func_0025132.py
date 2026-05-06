def max_neighbor(self, in_lon, in_lat, radius=0.05):
        """
        Finds the largest value within a given radius of a point on the interpolated grid.

        Args:
            in_lon: 2D array of longitude values
            in_lat: 2D array of latitude values
            radius: radius of influence for largest neighbor search in degrees

        Returns:
            Array of interpolated data
        """
        out_data = np.zeros((self.data.shape[0], in_lon.shape[0], in_lon.shape[1]))
        in_tree = cKDTree(np.vstack((in_lat.ravel(), in_lon.ravel())).T)
        out_indices = np.indices(out_data.shape[1:])
        out_rows = out_indices[0].ravel()
        out_cols = out_indices[1].ravel()
        for d in range(self.data.shape[0]):
            nz_points = np.where(self.data[d] > 0)
            if len(nz_points[0]) > 0:
                nz_vals = self.data[d][nz_points]
                nz_rank = np.argsort(nz_vals)
                original_points = cKDTree(np.vstack((self.lat[nz_points[0][nz_rank]], self.lon[nz_points[1][nz_rank]])).T)
                all_neighbors = original_points.query_ball_tree(in_tree, radius, p=2, eps=0)
                for n, neighbors in enumerate(all_neighbors):
                    if len(neighbors) > 0:
                        out_data[d, out_rows[neighbors], out_cols[neighbors]] = nz_vals[nz_rank][n]
        return out_data