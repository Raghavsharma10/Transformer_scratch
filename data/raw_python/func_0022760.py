def conjugate(self):
        """ Obtain the conjugate of the quaternion.
        
        This is simply the same quaternion but with the sign of the
        imaginary (vector) parts reversed.
        """
        new = self.copy()
        new.x *= -1
        new.y *= -1
        new.z *= -1
        return new