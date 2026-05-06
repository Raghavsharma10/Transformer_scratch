def from_parameters3(cls, lengths, angles):
        """Construct a 3D unit cell with the given parameters

           The a vector is always parallel with the x-axis and they point in the
           same direction. The b vector is always in the xy plane and points
           towards the positive y-direction. The c vector points towards the
           positive z-direction.
        """
        for length in lengths:
            if length <= 0:
                raise ValueError("The length parameters must be strictly positive.")
        for angle in angles:
            if angle <= 0 or angle >= np.pi:
                raise ValueError("The angle parameters must lie in the range ]0 deg, 180 deg[.")
        del length
        del angle

        matrix = np.zeros((3, 3), float)

        # first cell vector along x-axis
        matrix[0, 0] = lengths[0]

        # second cell vector in x-y plane
        matrix[0, 1] = np.cos(angles[2])*lengths[1]
        matrix[1, 1] = np.sin(angles[2])*lengths[1]

        # Finding the third cell vector is slightly more difficult. :-)
        # It works like this:
        # The dot products of a with c, b with c and c with c are known. the
        # vector a has only an x component, b has no z component. This results
        # in the following equations:
        u_a = lengths[0]*lengths[2]*np.cos(angles[1])
        u_b = lengths[1]*lengths[2]*np.cos(angles[0])
        matrix[0, 2] = u_a/matrix[0, 0]
        matrix[1, 2] = (u_b - matrix[0, 1]*matrix[0, 2])/matrix[1, 1]
        u_c = lengths[2]**2 - matrix[0, 2]**2 - matrix[1, 2]**2
        if u_c < 0:
            raise ValueError("The given cell parameters do not correspond to a unit cell.")
        matrix[2, 2] = np.sqrt(u_c)

        active = np.ones(3, bool)
        return cls(matrix, active)