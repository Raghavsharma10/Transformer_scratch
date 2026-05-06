def freeze(self):
        """
        Freeze (disable) all settings so they can't be altered
        """
        for xsl, xsr, ys, nx, ny in \
                zip(self.xsl, self.xsr,
                    self.ys, self.nx, self.ny):
            xsl.disable()
            xsr.disable()
            ys.disable()
            nx.disable()
            ny.disable()
        self.npair.disable()
        self.xbin.disable()
        self.ybin.disable()
        self.sbutt.disable()
        self.frozen = True