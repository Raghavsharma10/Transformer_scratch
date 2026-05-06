def create_from_axis_angle(cls, angle, ax, ay, az, degrees=False):
        """ Classmethod to create a quaternion from an axis-angle representation. 
        (angle should be in radians).
        """
        if degrees:
            angle = np.radians(angle)
        while angle < 0:
            angle += np.pi*2
        angle2 = angle/2.0
        sinang2 = np.sin(angle2)
        return Quaternion(np.cos(angle2), ax*sinang2, ay*sinang2, az*sinang2)