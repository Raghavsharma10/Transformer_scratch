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

        if any(self.chi_1 < -1.0) or any(self.chi_1 > 1.0):
            raise ValueError("Chi 1 is outside [-1.0, 1.0].")

        if any(self.chi_2 < -1.0) or any(self.chi_2 > 1.0):
            raise ValueError("Chi 2 is outside [-1.0, 1.0].")

        if any(self.z <= 0.0):
            raise ValueError("Redshift is zero or negative.")

        if any(self.dist <= 0.0):
            raise ValueError("Distance is zero or negative.")

        if any(self.st < 0.0):
            raise ValueError("Start Time is negative.")

        if any(self.et < 0.0):
            raise ValueError("End Time is negative.")

        if len(np.where(self.st < self.et)[0]) != 0:
            raise ValueError("Start Time is less than End time.")

        if any(self.m1/self.m2 > 1.0000001e4) or any(self.m1/self.m2 < 9.999999e-5):
            raise ValueError("Mass Ratio too far from unity.")

        return