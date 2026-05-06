def _intersect_edge_arrays(self, lines1, lines2):
        """Return the intercepts of all lines defined in *lines1* as they 
        intersect all lines in *lines2*. 
        
        Arguments are of shape (..., 2, 2), where axes are:
        
        0: number of lines
        1: two points per line
        2: x,y pair per point

        Lines are compared elementwise across the arrays (lines1[i] is compared
        against lines2[i]). If one of the arrays has N=1, then that line is
        compared against all lines in the other array.
        
        Returns an array of shape (N,) where each value indicates the intercept
        relative to the defined line segment. A value of 0 indicates 
        intersection at the first endpoint, and a value of 1 indicates 
        intersection at the second endpoint. Values between 1 and 0 are on the
        segment, whereas values outside 1 and 0 are off of the segment. 
        """
        # vector for each line in lines1
        l1 = lines1[..., 1, :] - lines1[..., 0, :]
        # vector for each line in lines2
        l2 = lines2[..., 1, :] - lines2[..., 0, :]
        # vector between first point of each line
        diff = lines1[..., 0, :] - lines2[..., 0, :]
        
        p = l1.copy()[..., ::-1]  # vectors perpendicular to l1
        p[..., 0] *= -1
        
        f = (l2 * p).sum(axis=-1)  # l2 dot p
        # tempting, but bad idea! 
        #f = np.where(f==0, 1, f)
        err = np.geterr()
        np.seterr(divide='ignore', invalid='ignore')
        try:
            h = (diff * p).sum(axis=-1) / f  # diff dot p / f
        finally:
            np.seterr(**err)
        
        return h