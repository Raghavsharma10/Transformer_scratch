def create_from_euler_angles(cls, rx, ry, rz, degrees=False):
        """ Classmethod to create a quaternion given the euler angles.
        """
        if degrees:
            rx, ry, rz = np.radians([rx, ry, rz])
        # Obtain quaternions
        qx = Quaternion(np.cos(rx/2), 0, 0, np.sin(rx/2))
        qy = Quaternion(np.cos(ry/2), 0, np.sin(ry/2), 0)
        qz = Quaternion(np.cos(rz/2), np.sin(rz/2), 0, 0)
        # Almost done
        return qx*qy*qz