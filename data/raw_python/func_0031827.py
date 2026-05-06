def calc_lfp_layer(self):
        """
        Calculate the LFP from concatenated subpopulations residing in a
        certain layer, e.g all L4E pops are summed, according to the `mapping_Yy`
        attribute of the `hybridLFPy.Population` objects.
        """
        LFPdict = {}

        lastY = None
        for Y, y in self.mapping_Yy:
            if lastY != Y:
                try:
                    LFPdict.update({Y : self.LFPdict[y]})
                except KeyError:
                    pass
            else:
                try:
                    LFPdict[Y] += self.LFPdict[y]
                except KeyError:
                    pass
            lastY = Y

        return LFPdict