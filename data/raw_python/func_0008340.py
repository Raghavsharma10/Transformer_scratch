def freeze(self):
        """
        Freeze (disable) all settings
        """
        for fields in zip(self.xsll, self.xsul, self.xslr, self.xsur,
                          self.ys, self.nx, self.ny):
            for field in fields:
                field.disable()
        self.nquad.disable()
        self.xbin.disable()
        self.ybin.disable()
        self.sbutt.disable()
        self.frozen = True