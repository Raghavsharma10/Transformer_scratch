def get_axis_angle(self):
        """ Get the axis-angle representation of the quaternion. 
        (The angle is in radians)
        """
        # Init
        angle = 2 * np.arccos(max(min(self.w, 1.), -1.))
        scale = (self.x**2 + self.y**2 + self.z**2)**0.5    
        
        # Calc axis
        if scale:
            ax = self.x / scale
            ay = self.y / scale
            az = self.z / scale
        else:
            # No rotation, so arbitrary axis
            ax, ay, az = 1, 0, 0
        # Return
        return angle, ax, ay, az