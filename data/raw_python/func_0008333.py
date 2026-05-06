def check(self):
        """
        Checks the values of the window pairs. If any problems are found, it
        flags them by changing the background colour.

        Returns (status, synced)

          status : flag for whether parameters are viable at all
          synced : flag for whether the windows are synchronised.
        """

        status = True
        synced = True

        xbin = self.xbin.value()
        ybin = self.ybin.value()
        npair = self.npair.value()

        g = get_root(self).globals
        # individual pair checks
        for xslw, xsrw, ysw, nxw, nyw in zip(self.xsl[:npair], self.xsr[:npair],
                                             self.ys[:npair], self.nx[:npair],
                                             self.ny[:npair]):
            xslw.config(bg=g.COL['main'])
            xsrw.config(bg=g.COL['main'])
            ysw.config(bg=g.COL['main'])
            nxw.config(bg=g.COL['main'])
            nyw.config(bg=g.COL['main'])
            status = status if xslw.ok() else False
            status = status if xsrw.ok() else False
            status = status if ysw.ok() else False
            status = status if nxw.ok() else False
            status = status if nyw.ok() else False
            xsl = xslw.value()
            xsr = xsrw.value()
            ys = ysw.value()
            nx = nxw.value()
            ny = nyw.value()

            # Are unbinned dimensions consistent with binning factors?
            if nx is None or nx % xbin != 0:
                nxw.config(bg=g.COL['error'])
                status = False
            elif (nx // xbin) % 4 != 0:
                """
                The NGC collects pixel data in chunks before transmission.
                As a result, to avoid loss of data from frames, the binned
                x-size must be a multiple of 4.
                """
                nxw.config(bg=g.COL['error'])
                status = False

            if ny is None or ny % ybin != 0:
                nyw.config(bg=g.COL['error'])
                status = False

            # overlap checks
            if xsl is None or xsr is None or xsl >= xsr:
                xsrw.config(bg=g.COL['error'])
                status = False

            if xsl is None or xsr is None or nx is None or xsl + nx > xsr:
                xsrw.config(bg=g.COL['error'])
                status = False

            # Are the windows synchronised? This means that they would
            # be consistent with the pixels generated were the whole CCD
            # to be binned by the same factors. If relevant values are not
            # set, we count that as "synced" because the purpose of this is
            # to enable / disable the sync button and we don't want it to be
            # enabled just because xs or ys are not set.
            perform_check = all([param is not None for param in (xsl, xsr, ys, nx, ny)])
            if (perform_check and
                ((xsl - 1) % xbin != 0 or (xsr - 1025) % xbin != 0 or
                 (ys - 1) % ybin != 0)):
                synced = False

            # Range checks
            if xsl is None or nx is None or xsl + nx - 1 > xslw.imax:
                xslw.config(bg=g.COL['error'])
                status = False

            if xsr is None or nx is None or xsr + nx - 1 > xsrw.imax:
                xsrw.config(bg=g.COL['error'])
                status = False

            if ys is None or ny is None or ys + ny - 1 > ysw.imax:
                ysw.config(bg=g.COL['error'])
                status = False

        # Pair overlap checks. Compare one pair with the next one in the
        # same quadrant (if there is one). Only bother if we have survived
        # so far, which saves a lot of checks
        if status:
            for index in range(npair-2):
                ys1 = self.ys[index].value()
                ny1 = self.ny[index].value()

                ysw2 = self.ys[index+2]
                ys2 = ysw2.value()
                if ys1 + ny1 > ys2:
                    ysw2.config(bg=g.COL['error'])
                    status = False

        if synced:
            self.sbutt.config(bg=g.COL['main'])
            self.sbutt.disable()
        else:
            if not self.frozen:
                self.sbutt.enable()
            self.sbutt.config(bg=g.COL['warn'])

        return status