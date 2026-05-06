def cylindrical(cls, mag, theta, z=0):
        '''Returns a Vector instance from cylindircal coordinates'''
        return cls(
            mag * math.cos(theta),  # X
            mag * math.sin(theta),  # Y
            z  # Z
        )