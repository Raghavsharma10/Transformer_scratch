def _central_slopes_directions(self, data, dX, dY):
        """
        Calculates magnitude/direction of slopes using central difference
        """
        shp = np.array(data.shape) - 1

        direction = np.full(data.shape, FLAT_ID_INT, 'float64')
        mag = np.full(direction, FLAT_ID_INT, 'float64')

        ind = 0
        d1, d2, theta = _get_d1_d2(dX, dY, ind, [0, 1], [1, 1], shp)
        s2 = (data[0:-2, 1:-1] - data[2:, 1:-1]) / d2
        s1 = -(data[1:-1, 0:-2] - data[1:-1, 2:]) / d1
        direction[1:-1, 1:-1] = np.arctan2(s2, s1) + np.pi
        mag = np.sqrt(s1**2 + s2**2)

        return mag, direction