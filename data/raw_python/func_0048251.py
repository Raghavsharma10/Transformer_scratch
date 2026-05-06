def calc_ABDE_from_lamination_parameters(self):
        """Use the ABDE matrix based on lamination parameters.

        Given the lamination parameters ``xiA``, ``xiB``, ``xiC`` and ``xiD``,
        the ABD matrix is calculated.

        """
        # dummies used to unpack vector results
        du1, du2, du3, du4, du5, du6 = 0, 0, 0, 0, 0, 0
        # A matrix terms
        A11,A22,A12, du1,du2,du3, A66,A16,A26 =\
            (self.h       ) * np.dot(self.matobj.u, self.xiA)
        # B matrix terms
        B11,B22,B12, du1,du2,du3, B66,B16,B26 =\
            (self.h**2/4. ) * np.dot(self.matobj.u, self.xiB)
        # D matrix terms
        D11,D22,D12, du1,du2,du3, D66,D16,D26 =\
            (self.h**3/12.) * np.dot(self.matobj.u, self.xiD)
        # E matrix terms
        du1,du2,du3, E44,E55,E45, du4,du5,du6 =\
            (self.h       ) * np.dot(self.matobj.u, self.xiE)

        self.A = np.array([[A11, A12, A16],
                           [A12, A22, A26],
                           [A16, A26, A66]], dtype=np.float64)

        self.B = np.array([[B11, B12, B16],
                           [B12, B22, B26],
                           [B16, B26, B66]], dtype=np.float64)

        self.D = np.array([[D11, D12, D16],
                           [D12, D22, D26],
                           [D16, D26, D66]], dtype=np.float64)

        # printing E acoordingly to Reddy definition for E44, E45 and E55
        self.E = np.array([[E55, E45],
                           [E45, E44]], dtype=np.float64)

        self.ABD = np.array([[A11, A12, A16, B11, B12, B16],
                             [A12, A22, A26, B12, B22, B26],
                             [A16, A26, A66, B16, B26, B66],
                             [B11, B12, B16, D11, D12, D16],
                             [B12, B22, B26, D12, D22, D26],
                             [B16, B26, B66, D16, D26, D66]], dtype=np.float64)

        # printing ABDE acoordingly to Reddy definition for E44, E45 and E55
        self.ABDE = np.array([[A11, A12, A16, B11, B12, B16, 0, 0],
                              [A12, A22, A26, B12, B22, B26, 0, 0],
                              [A16, A26, A66, B16, B26, B66, 0, 0],
                              [B11, B12, B16, D11, D12, D16, 0, 0],
                              [B12, B22, B26, D12, D22, D26, 0, 0],
                              [B16, B26, B66, D16, D26, D66, 0, 0],
                              [0, 0, 0, 0, 0, 0, E55, E45],
                              [0, 0, 0, 0, 0, 0, E45, E44]],
                               dtype=np.float64)