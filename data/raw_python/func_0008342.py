def check(self):
        """
        Checks the values of the windows. If any problems are found,
        it flags them by changing the background colour. Only active
        windows are checked.

        Returns status, flag for whether parameters are viable.
        """

        status = True
        synced = True

        xbin = self.xbin.value()
        ybin = self.ybin.value()
        nwin = self.nwin.value()

        # individual window checks
        g = get_root(self).globals
        for xsw, ysw, nxw, nyw in \
                zip(self.xs[:nwin], self.ys[:nwin],
                    self.nx[:nwin], self.ny[:nwin]):

            xsw.config(bg=g.COL['main'])
            ysw.config(bg=g.COL['main'])
            nxw.config(bg=g.COL['main'])
            nyw.config(bg=g.COL['main'])
            status = status if xsw.ok() else False
            status = status if ysw.ok() else False
            status = status if nxw.ok() else False
            status = status if nyw.ok() else False
            xs = xsw.value()
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

            # Are the windows synchronised? This means that they
            # would be consistent with the pixels generated were
            # the whole CCD to be binned by the same factors
            # If relevant values are not set, we count that as
            # "synced" because the purpose of this is to enable
            # / disable the sync button and we don't want it to be
            # enabled just because xs or ys are not set.
            if (xs is not None and ys is not None and nx is not None and
                ny is not None):
                    if (xs < 1025 and ((xs - 1) % xbin != 0 or (ys - 1) % ybin != 0)
                        or ((xs-1025) % xbin != 0 or (ys - 1) % ybin != 0)):
                        synced = False

            # Range checks
            if xs is None or nx is None or xs + nx - 1 > xsw.imax:
                xsw.config(bg=g.COL['error'])
                status = False

            if ys is None or ny is None or ys + ny - 1 > ysw.imax:
                ysw.config(bg=g.COL['error'])
                status = False

        # Overlap checks. Compare each window with the next one, requiring
        # no y overlap and that the second is higher than the first
        if status:
            n1 = 0
            for ysw1, nyw1 in zip(self.ys[:nwin-1], self.ny[:nwin-1]):

                ys1 = ysw1.value()
                ny1 = nyw1.value()

                n1 += 1
                ysw2 = self.ys[n1]

                ys2 = ysw2.value()

                if ys2 < ys1 + ny1:
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