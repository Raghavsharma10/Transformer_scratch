def _sanity_check(self):
        """Check if parameters are okay.

        Sanity check makes sure each parameter is within an allowable range.

        Raises:
            ValueError: Problem with a specific parameter.

        """
        if any(self.m1 < 0.0):
            raise ValueError("Mass 1 is negative.")
        if any(self.m2 < 0.0):
            raise ValueError("Mass 2 is negative.")

        if any(self.z <= 0.0):
            raise ValueError("Redshift is zero or negative.")

        if any(self.dist <= 0.0):
            raise ValueError("Distance is zero or negative.")

        if any(self.initial_point < 0.0):
            raise ValueError("initial_point is negative.")

        if any(self.t_obs < 0.0):
            raise ValueError("t_obs is negative.")

        if any(self.e0 <= 0.0):
            raise ValueError("e0 must be greater than zero when using EccentricBinaries class.")

        if any(self.e0 > 1.0):
            raise ValueError("e0 greater than 1.")

        return