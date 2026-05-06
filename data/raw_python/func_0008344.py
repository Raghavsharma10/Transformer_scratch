def freeze(self):
        """
        Freeze all settings so they can't be altered
        """
        for xs, ys, nx, ny in \
                zip(self.xs, self.ys, self.nx, self.ny):
            xs.disable()
            ys.disable()
            nx.disable()
            ny.disable()
        self.nwin.disable()
        self.xbin.disable()
        self.ybin.disable()
        self.sbutt.disable()
        self.frozen = True