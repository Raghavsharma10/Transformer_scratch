def sync(self, *args):
        """
        Synchronise the settings. This means that the pixel start
        values are shifted downwards so that they are synchronised
        with a full-frame binned version. This does nothing if the
        binning factor == 1
        """
        xbin = self.xbin.value()
        ybin = self.ybin.value()
        n = 0
        for xs, ys, nx, ny in self:
            if xbin > 1 and xs % xbin != 1:
                if xs < 1025:
                    xs = xbin*((xs-1)//xbin)+1
                else:
                    xs = xbin*((xs-1025)//xbin)+1025
                self.xs[n].set(xs)

            if ybin > 1 and ys % ybin != 1:
                ys = ybin*((ys-1)//ybin)+1
                self.ys[n].set(ys)

            n += 1
        self.sbutt.config(bg=g.COL['main'])
        self.sbutt.config(state='disable')