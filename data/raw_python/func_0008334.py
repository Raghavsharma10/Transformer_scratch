def sync(self):
        """
        Synchronise the settings. This means that the pixel start
        values are shifted downwards so that they are synchronised
        with a full-frame binned version. This does nothing if the
        binning factors == 1.
        """

        # needs some mods for ultracam ??
        xbin = self.xbin.value()
        ybin = self.ybin.value()
        n = 0
        for xsl, xsr, ys, nx, ny in self:
            if xbin > 1:
                xsl = xbin*((xsl-1)//xbin)+1
                self.xsl[n].set(xsl)
                xsr = xbin*((xsr-1025)//xbin)+1025
                self.xsr[n].set(xsr)

            if ybin > 1:
                ys = ybin*((ys-1)//ybin)+1
                self.ys[n].set(ys)

            n += 1
        g = get_root(self).globals
        self.sbutt.config(bg=g.COL['main'])
        self.sbutt.config(state='disable')