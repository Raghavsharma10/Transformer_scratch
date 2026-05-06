def write_reg(self, filename):
        """
        Write a ds9 region file that represents this region as a set of diamonds.

        Parameters
        ----------
        filename : str
            File to write
        """
        with open(filename, 'w') as out:
            for d in range(1, self.maxdepth+1):
                for p in self.pixeldict[d]:
                    line = "fk5; polygon("
                    # the following int() gets around some problems with np.int64 that exist prior to numpy v 1.8.1
                    vectors = list(zip(*hp.boundaries(2**d, int(p), step=1, nest=True)))
                    positions = []
                    for sky in self.vec2sky(np.array(vectors), degrees=True):
                        ra, dec = sky
                        pos = SkyCoord(ra/15, dec, unit=(u.degree, u.degree))
                        positions.append(pos.ra.to_string(sep=':', precision=2))
                        positions.append(pos.dec.to_string(sep=':', precision=2))
                    line += ','.join(positions)
                    line += ")"
                    print(line, file=out)
        return