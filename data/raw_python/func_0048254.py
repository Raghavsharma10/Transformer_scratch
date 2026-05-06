def force_orthotropic(self):
        r"""Force an orthotropic laminate

        The terms
        `A_{13}`, `A_{23}`, `A_{31}`, `A_{32}`,
        `B_{13}`, `B_{23}`, `B_{31}`, `B_{32}`,
        `D_{13}`, `D_{23}`, `D_{31}`, `D_{32}` are set to zero to force an
        orthotropic laminate.

        """
        if self.offset != 0.:
            raise RuntimeError(
                    'Laminates with offset cannot be forced orthotropic!')
        self.A[0, 2] = 0.
        self.A[1, 2] = 0.
        self.A[2, 0] = 0.
        self.A[2, 1] = 0.

        self.B[0, 2] = 0.
        self.B[1, 2] = 0.
        self.B[2, 0] = 0.
        self.B[2, 1] = 0.

        self.D[0, 2] = 0.
        self.D[1, 2] = 0.
        self.D[2, 0] = 0.
        self.D[2, 1] = 0.

        self.ABD[0, 2] = 0. # A16
        self.ABD[1, 2] = 0. # A26
        self.ABD[2, 0] = 0. # A61
        self.ABD[2, 1] = 0. # A62

        self.ABD[0, 5] = 0. # B16
        self.ABD[5, 0] = 0. # B61
        self.ABD[1, 5] = 0. # B26
        self.ABD[5, 1] = 0. # B62

        self.ABD[3, 2] = 0. # B16
        self.ABD[2, 3] = 0. # B61
        self.ABD[4, 2] = 0. # B26
        self.ABD[2, 4] = 0. # B62

        self.ABD[3, 5] = 0. # D16
        self.ABD[4, 5] = 0. # D26
        self.ABD[5, 3] = 0. # D61
        self.ABD[5, 4] = 0. # D62

        self.ABDE[0, 2] = 0. # A16
        self.ABDE[1, 2] = 0. # A26
        self.ABDE[2, 0] = 0. # A61
        self.ABDE[2, 1] = 0. # A62

        self.ABDE[0, 5] = 0. # B16
        self.ABDE[5, 0] = 0. # B61
        self.ABDE[1, 5] = 0. # B26
        self.ABDE[5, 1] = 0. # B62

        self.ABDE[3, 2] = 0. # B16
        self.ABDE[2, 3] = 0. # B61
        self.ABDE[4, 2] = 0. # B26
        self.ABDE[2, 4] = 0. # B62

        self.ABDE[3, 5] = 0. # D16
        self.ABDE[4, 5] = 0. # D26
        self.ABDE[5, 3] = 0. # D61
        self.ABDE[5, 4] = 0.