def calc_equivalent_modulus(self):
        """Calculates the equivalent laminate properties.

        The following attributes are calculated:
            e1, e2, g12, nu12, nu21

        """
        AI = np.matrix(self.ABD, dtype=np.float64).I
        a11, a12, a22, a33 = AI[0,0], AI[0,1], AI[1,1], AI[2,2]
        self.e1 = 1./(self.h*a11)
        self.e2 = 1./(self.h*a22)
        self.g12 = 1./(self.h*a33)
        self.nu12 = - a12 / a11
        self.nu21 = - a12 / a22