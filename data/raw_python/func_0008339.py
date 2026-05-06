def sync(self):
        """
        Synchronise the settings.

        This routine changes the window settings so that the pixel start
        values are shifted downwards until they are synchronised with a
        full-frame binned version. This does nothing if the binning factor
        is 1.
        """
        xbin = self.xbin.value()
        ybin = self.ybin.value()
        if xbin == 1 and ybin == 1:
            self.sbutt.config(state='disable')
            return

        for n, (xsll, xsul, xslr, xsur, ys, nx, ny) in enumerate(self):
            if (xsll-1) % xbin != 0:
                xsll = xbin * ((xsll-1)//xbin)+1
                self.xsll[n].set(xsll)
            if (xsul-1) % xbin != 0:
                xsul = xbin * ((xsul-1)//xbin)+1
                self.xsul[n].set(xsul)
            if (xslr-1025) % xbin != 0:
                xslr = xbin * ((xslr-1025)//xbin)+1025
                self.xslr[n].set(xslr)
            if (xsur-1025) % xbin != 0:
                xsur = xbin * ((xsur-1025)//xbin)+1025
                self.xsur[n].set(xsur)

            if ybin > 1 and (ys-1) % ybin != 0:
                ys = ybin*((ys-1)//ybin)+1
                self.ys[n].set(ys)

        self.sbutt.config(bg=g.COL['main'])
        self.sbutt.config(state='disable')