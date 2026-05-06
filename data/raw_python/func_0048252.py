def calc_constitutive_matrix(self):
        """Calculates the laminate constitutive matrix

        This is the commonly called ``ABD`` matrix with ``shape=(6, 6)`` when
        the classical laminated plate theory is used, or the ``ABDE`` matrix
        when the first-order shear deformation theory is used, containing the
        transverse shear terms.

        """
        self.A_general = np.zeros([5,5], dtype=np.float64)
        self.B_general = np.zeros([5,5], dtype=np.float64)
        self.D_general = np.zeros([5,5], dtype=np.float64)

        lam_thick = sum([ply.h for ply in self.plies])
        self.h = lam_thick

        h0 = -lam_thick/2 + self.offset
        for ply in self.plies:
            hk_1 = h0
            h0 += ply.h
            hk = h0
            self.A_general += ply.QL*(hk - hk_1)
            self.B_general += 1/2.*ply.QL*(hk**2 - hk_1**2)
            self.D_general += 1/3.*ply.QL*(hk**3 - hk_1**3)

        self.A = self.A_general[0:3, 0:3]
        self.B = self.B_general[0:3, 0:3]
        self.D = self.D_general[0:3, 0:3]
        self.E = self.A_general[3:5, 3:5]

        conc1 = np.concatenate([self.A, self.B], axis=1)
        conc2 = np.concatenate([self.B, self.D], axis=1)

        self.ABD = np.concatenate([conc1, conc2], axis=0)
        self.ABDE = np.zeros((8, 8), dtype=np.float64)
        self.ABDE[0:6, 0:6] = self.ABD
        self.ABDE[6:8, 6:8] = self.E