def getRaDecs(self, mods):
        """Internal function converting cartesian coords to
        ra dec"""
        raDecOut = np.empty( (len(mods), 5))
        raDecOut[:,0:3] = mods[:,0:3]

        for i, row in enumerate(mods):
            raDecOut[i, 3:5] = r.raDecFromVec(row[3:6])
        return raDecOut