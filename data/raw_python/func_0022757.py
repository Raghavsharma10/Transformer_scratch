def copy(self):
        """ Create an exact copy of this quaternion. 
        """
        return Quaternion(self.w, self.x, self.y, self.z, False)