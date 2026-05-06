def _intersection_matrix(self, lines):
        """
        Return a 2D array of intercepts such that 
        intercepts[i, j] is the intercept of lines[i] onto lines[j].
        
        *lines* must be an array of point locations with shape (N, 2, 2), where
        the axes are (lines, points_per_line, xy_per_point).
        
        The intercept is described in intersect_edge_arrays().
        """
        return self._intersect_edge_arrays(lines[:, np.newaxis, ...], 
                                           lines[np.newaxis, ...])