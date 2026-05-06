def enable(self):
        """
        Enables WinQuad setting
        """
        nquad = self.nquad.value()
        for label, xsll, xsul, xslr, xsur, ys, nx, ny in \
                zip(self.label[:nquad], self.xsll[:nquad], self.xsul[:nquad],
                    self.xslr[:nquad], self.xsur[:nquad], self.ys[:nquad],
                    self.nx[:nquad], self.ny[:nquad]):
            label.config(state='normal')
            for thing in (xsll, xsul, xslr, xsur, ys, nx, ny):
                thing.enable()

        for label, xsll, xsul, xslr, xsur, ys, nx, ny in \
                zip(self.label[nquad:], self.xsll[nquad:], self.xsul[nquad:],
                    self.xslr[nquad:], self.xsur[nquad:], self.ys[nquad:],
                    self.nx[nquad:], self.ny[nquad:]):
            label.config(state='disable')
            for thing in (xsll, xsul, xslr, xsur, ys, nx, ny):
                thing.disable()

        self.nquad.enable()
        self.xbin.enable()
        self.ybin.enable()
        self.sbutt.enable()