def cast(cls, c):
        """Convert the first argument into a Complete object"""
        if isinstance(c, Complete):
            return c
        elif isinstance(c, Translation):
            return Complete(np.identity(3, float), c.t)
        elif isinstance(c, Rotation):
            return Complete(c.r, np.zeros(3, float))