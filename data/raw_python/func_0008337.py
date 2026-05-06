def enable(self):
        """
        Enables WinPair settings
        """
        npair = self.npair.value()
        for label, xsl, xsr, ys, nx, ny in \
                zip(self.label[:npair], self.xsl[:npair], self.xsr[:npair],
                    self.ys[:npair], self.nx[:npair], self.ny[:npair]):
            label.config(state='normal')
            xsl.enable()
            xsr.enable()
            ys.enable()
            nx.enable()
            ny.enable()

        for label, xsl, xsr, ys, nx, ny in \
                zip(self.label[npair:], self.xsl[npair:], self.xsr[npair:],
                    self.ys[npair:], self.nx[npair:], self.ny[npair:]):
            label.config(state='disable')
            xsl.disable()
            xsr.disable()
            ys.disable()
            nx.disable()
            ny.disable()

        self.npair.enable()
        self.xbin.enable()
        self.ybin.enable()
        self.sbutt.enable()