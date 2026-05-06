def set_mapping(self, points1, points2):
        """ Set to a 3D transformation matrix that maps points1 onto points2.

        Parameters
        ----------
        points1 : array-like, shape (4, 3)
            Four starting 3D coordinates.
        points2 : array-like, shape (4, 3)
            Four ending 3D coordinates.
        """
        # note: need to transpose because util.functions uses opposite
        # of standard linear algebra order.
        self.matrix = transforms.affine_map(points1, points2).T