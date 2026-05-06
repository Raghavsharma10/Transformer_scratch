def norm(self):
        """ Returns the norm of the quaternion
        
        norm = w**2 + x**2 + y**2 + z**2
        """
        tmp = self.w**2 + self.x**2 + self.y**2 + self.z**2
        return tmp**0.5