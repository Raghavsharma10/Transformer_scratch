def log(self):
        """ Returns the natural logarithm of the quaternion. 
        (not tested)
        """
        
        # Init
        norm = self.norm()
        vecNorm = self.x**2 + self.y**2 + self.z**2
        tmp = self.w / norm
        q = Quaternion()
        
        # Calculate
        q.w = np.log(norm)
        q.x = np.log(norm) * self.x * np.arccos(tmp) / vecNorm
        q.y = np.log(norm) * self.y * np.arccos(tmp) / vecNorm
        q.z = np.log(norm) * self.z * np.arccos(tmp) / vecNorm
        
        return q