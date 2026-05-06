def properties(self):
        """Rotation properties: angle, axis, invert"""
        # determine wether an inversion rotation has been applied
        invert = (np.linalg.det(self.r) < 0)
        factor = {True: -1, False: 1}[invert]
        # get the rotation data
        # trace(r) = 1+2*cos(angle)
        cos_angle = 0.5*(factor*np.trace(self.r) - 1)
        if cos_angle > 1: cos_angle = 1.0
        if cos_angle < -1: cos_angle = -1.0
        # the antisymmetric part of the non-diagonal vector tell us something
        # about sin(angle) and n.
        axis = 0.5*factor*np.array([-self.r[1, 2] + self.r[2, 1], self.r[0, 2] - self.r[2, 0], -self.r[0, 1] + self.r[1, 0]])
        sin_angle = np.linalg.norm(axis)
        # look for the best way to normalize the
        if (sin_angle == 0.0) and (cos_angle > 0):
            axis[2] = 1.0
        elif abs(sin_angle) < (1-cos_angle):
            for index in range(3):
                axis[index] = {True: -1, False: 1}[axis[index] < 0] * np.sqrt(abs((factor*self.r[index, index] - cos_angle) / (1 - cos_angle)))
        else:
            axis = axis / sin_angle

        # Finally calculate the angle:
        angle = np.arctan2(sin_angle, cos_angle)
        return angle, axis, invert