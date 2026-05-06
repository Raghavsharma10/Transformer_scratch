def calc_lamination_parameters(self):
        """Calculate the lamination parameters.

        The following attributes are calculated:
            xiA, xiB, xiD, xiE

        """
        if len(self.plies) == 0:
            if self.xiA is None:
                raise ValueError('Laminate with 0 plies!')
            else:
                return
        xiA1, xiA2, xiA3, xiA4 = 0, 0, 0, 0
        xiB1, xiB2, xiB3, xiB4 = 0, 0, 0, 0
        xiD1, xiD2, xiD3, xiD4 = 0, 0, 0, 0
        xiE1, xiE2, xiE3, xiE4 = 0, 0, 0, 0

        lam_thick = sum([ply.h for ply in self.plies])
        self.h = lam_thick

        h0 = -lam_thick/2. + self.offset
        for ply in self.plies:
            if self.matobj is None:
                self.matobj = ply.matobj
            else:
                assert np.allclose(self.matobj.u, ply.matobj.u), "Plies with different materials"
            hk_1 = h0
            h0 += ply.h
            hk = h0

            Afac = ply.h / lam_thick
            Bfac = (2. / lam_thick**2) * (hk**2 - hk_1**2)
            Dfac = (4. / lam_thick**3) * (hk**3 - hk_1**3)
            Efac = (1. / lam_thick) * (hk - hk_1)

            thetarad = np.deg2rad(ply.theta)
            cos2t = np.cos(2*thetarad)
            sin2t = np.sin(2*thetarad)
            cos4t = np.cos(4*thetarad)
            sin4t = np.sin(4*thetarad)

            xiA1 += Afac * cos2t
            xiA2 += Afac * sin2t
            xiA3 += Afac * cos4t
            xiA4 += Afac * sin4t

            xiB1 += Bfac * cos2t
            xiB2 += Bfac * sin2t
            xiB3 += Bfac * cos4t
            xiB4 += Bfac * sin4t

            xiD1 += Dfac * cos2t
            xiD2 += Dfac * sin2t
            xiD3 += Dfac * cos4t
            xiD4 += Dfac * sin4t

            xiE1 += Efac * cos2t
            xiE2 += Efac * sin2t
            xiE3 += Efac * cos4t
            xiE4 += Efac * sin4t

        self.xiA = np.array([1, xiA1, xiA2, xiA3, xiA4], dtype=np.float64)
        self.xiB = np.array([0, xiB1, xiB2, xiB3, xiB4], dtype=np.float64)
        self.xiD = np.array([1, xiD1, xiD2, xiD3, xiD4], dtype=np.float64)
        self.xiE = np.array([1, xiE1, xiE2, xiE3, xiE4], dtype=np.float64)