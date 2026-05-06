def force_symmetric(self):
        """Force a symmetric laminate

        The `B` terms of the constitutive matrix are set to zero.

        """
        if self.offset != 0.:
            raise RuntimeError(
                    'Laminates with offset cannot be forced symmetric!')
        self.B = np.zeros((3,3))
        self.ABD[0:3, 3:6] = 0
        self.ABD[3:6, 0:3] = 0

        self.ABDE[0:3, 3:6] = 0
        self.ABDE[3:6, 0:3] = 0