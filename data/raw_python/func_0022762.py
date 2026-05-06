def exp(self):
        """ Returns the exponent of the quaternion. 
        (not tested)
        """
        
        # Init
        vecNorm = self.x**2 + self.y**2 + self.z**2
        wPart = np.exp(self.w)        
        q = Quaternion()
        
        # Calculate
        q.w = wPart * np.cos(vecNorm)
        q.x = wPart * self.x * np.sin(vecNorm) / vecNorm
        q.y = wPart * self.y * np.sin(vecNorm) / vecNorm
        q.z = wPart * self.z * np.sin(vecNorm) / vecNorm
        
        return q