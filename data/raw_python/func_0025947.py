def spherical(cls, mag, theta, phi=0):
        '''Returns a Vector instance from spherical coordinates'''
        return cls(
            mag * math.sin(phi) * math.cos(theta),  # X
            mag * math.sin(phi) * math.sin(theta),  # Y
            mag * math.cos(phi)  # Z
        )