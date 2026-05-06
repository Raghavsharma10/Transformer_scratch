def calc_csd_layer(self):
        """
        Calculate the CSD from concatenated subpopulations residing in a
        certain layer, e.g all L4E pops are summed, according to the `mapping_Yy`
        attribute of the `hybridLFPy.Population` objects.
        """
        CSDdict = {}

        lastY = None
        for Y, y in self.mapping_Yy:
            if lastY != Y:
                try:
                    CSDdict.update({Y : self.CSDdict[y]})
                except KeyError:
                    pass
            else:
                try:
                    CSDdict[Y] += self.CSDdict[y]
                except KeyError:
                    pass
            lastY = Y

        return CSDdict