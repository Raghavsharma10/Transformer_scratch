def computePointing(self, ra_deg, dec_deg, roll_deg, cartesian=False):
        """Compute a pointing model without changing the internal object pointing"""
        # Roll FOV
        Rrotate = r.rotateInXMat(roll_deg)  # Roll

        # Slew from ra/dec of zero
        Ra = r.rightAscensionRotationMatrix(ra_deg)
        Rd = r.declinationRotationMatrix(dec_deg)
        Rslew = np.dot(Ra, Rd)

        R = np.dot(Rslew, Rrotate)

        slew = self.origin*1
        for i, row in enumerate(self.origin):
            slew[i, 3:6] = np.dot(R, row[3:6])

        if cartesian is False:
            slew = self.getRaDecs(slew)
        return slew