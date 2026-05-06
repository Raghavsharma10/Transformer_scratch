def rotate_point(self, p):
        """ Rotate a Point instance using this quaternion.
        """
        # Prepare 
        p = Quaternion(0, p[0], p[1], p[2], False)  # Do not normalize!
        q1 = self.normalize()
        q2 = self.inverse()
        # Apply rotation
        r = (q1*p)*q2
        # Make point and return        
        return r.x, r.y, r.z