def random(cls):
        """Return a random rotation"""
        axis = random_unit()
        angle = np.random.uniform(0,2*np.pi)
        invert = bool(np.random.randint(0,2))
        return Rotation.from_properties(angle, axis, invert)