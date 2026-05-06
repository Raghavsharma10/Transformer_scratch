def calc_scf(self):
        """Calculate improved shear correction factors

        Reference:

            Vlachoutsis, S. "Shear correction factors for plates and shells",
            Int. Journal for Numerical Methods in Engineering, Vol. 33,
            1537-1552, 1992.

            http://onlinelibrary.wiley.com/doi/10.1002/nme.1620330712/full


        Using "one shear correction factor" (see reference), assuming:

        - constant G13, G23, E1, E2, nu12, nu21 within each ply
        - g1 calculated using z at the middle of each ply
        - zn1 = Laminate.offset

        Returns
        -------

        k13, k23 : tuple
            Shear correction factors. Also updates attributes: `scf_k13`
            and `scf_k23`.

        """
        D1 = 0
        R1 = 0
        den1 = 0

        D2 = 0
        R2 = 0
        den2 = 0

        offset = self.offset
        zbot = -self.h/2 + offset
        z1 = zbot

        for ply in self.plies:
            z2 = z1 + ply.h
            e1 = (ply.matobj.e1 * np.cos(np.deg2rad(ply.theta)) +
                  ply.matobj.e2 * np.sin(np.deg2rad(ply.theta)))
            e2 = (ply.matobj.e2 * np.cos(np.deg2rad(ply.theta)) +
                  ply.matobj.e1 * np.sin(np.deg2rad(ply.theta)))
            nu12 = (ply.matobj.nu12 * np.cos(np.deg2rad(ply.theta)) +
                  ply.matobj.nu21 * np.sin(np.deg2rad(ply.theta)))
            nu21 = (ply.matobj.nu21 * np.cos(np.deg2rad(ply.theta)) +
              ply.matobj.nu12 * np.sin(np.deg2rad(ply.theta)))

            D1 += e1 / (1 - nu12*nu21)
            R1 += D1*((z2 - offset)**3/3 - (z1 - offset)**3/3)
            g13 = ply.matobj.g13
            d1 = g13 * ply.h
            den1 += d1 * (self.h / ply.h) * D1**2*(15*offset*z1**4 + 30*offset*z1**2*zbot*(2*offset - zbot) - 15*offset*z2**4 + 30*offset*z2**2*zbot*(-2*offset + zbot) - 3*z1**5 + 10*z1**3*(-2*offset**2 - 2*offset*zbot + zbot**2) - 15*z1*zbot**2*(4*offset**2 - 4*offset*zbot + zbot**2) + 3*z2**5 + 10*z2**3*(2*offset**2 + 2*offset*zbot - zbot**2) + 15*z2*zbot**2*(4*offset**2 - 4*offset*zbot + zbot**2))/(60*g13)

            D2 += e2 / (1 - nu12*nu21)
            R2 += D2*((z2 - self.offset)**3/3 - (z1 - self.offset)**3/3)
            g23 = ply.matobj.g23
            d2 = g23 * ply.h
            den2 += d2 * (self.h / ply.h) * D2**2*(15*offset*z1**4 + 30*offset*z1**2*zbot*(2*offset - zbot) - 15*offset*z2**4 + 30*offset*z2**2*zbot*(-2*offset + zbot) - 3*z1**5 + 10*z1**3*(-2*offset**2 - 2*offset*zbot + zbot**2) - 15*z1*zbot**2*(4*offset**2 - 4*offset*zbot + zbot**2) + 3*z2**5 + 10*z2**3*(2*offset**2 + 2*offset*zbot - zbot**2) + 15*z2*zbot**2*(4*offset**2 - 4*offset*zbot + zbot**2))/(60*g23)

            z1 = z2

        self.scf_k13 = R1**2 / den1
        self.scf_k23 = R2**2 / den2

        return self.scf_k13, self.scf_k23