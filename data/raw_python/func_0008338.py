def check(self):
        """
        Checks the values of the window quads. If any problems are found it
        flags the offending window by changing the background colour.

        Returns:
            status : bool
        """
        status = synced = True

        xbin = self.xbin.value()
        ybin = self.ybin.value()
        nquad = self.nquad.value()

        g = get_root(self).globals
        # individual window checks
        for (xsllw, xsulw, xslrw, xsurw, ysw, nxw, nyw) in zip(
             self.xsll[:nquad],
             self.xsul[:nquad], self.xslr[:nquad],
             self.xsur[:nquad], self.ys[:nquad], self.nx[:nquad], self.ny[:nquad]):

            all_fields = (xsllw, xsulw, xslrw, xsurw, ysw, nxw, nyw)
            for field in all_fields:
                field.config(bg=g.COL['main'])
                status = status if field.ok() else False

            xsll = xsllw.value()
            xsul = xsulw.value()
            xslr = xslrw.value()
            xsur = xsurw.value()
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

            # overlap checks in x direction
            if xsll is None or xslr is None or xsll >= xslr:
                xslrw.config(bg=g.COL['error'])
                status = False
            if xsul is None or xsur is None or xsul >= xsur:
                xsurw.config(bg=g.COL['error'])
                status = False
            if nx is None or xsll is None or xsll + nx > xslr:
                xslrw.config(bg=g.COL['error'])
                status = False
            if xsul is None or nx is None or xsul + nx > xsur:
                xsurw.config(bg=g.COL['error'])
                status = False

            # Are the windows synchronised? This means that they would
            # be consistent with the pixels generated were the whole CCD
            # to be binned by the same factors. If relevant values are not
            # set, we count that as "synced" because the purpose of this is
            # to enable / disable the sync button and we don't want it to be
            # enabled just because xs or ys are not set.
            perform_check = all([param is not None for param in (
                xsll, xslr, ys, nx, ny
            )])
            if (perform_check and ((xsll - 1) % xbin != 0 or (xslr - 1025) % xbin != 0 or
                                   (ys - 1) % ybin != 0)):
                synced = False

            perform_check = all([param is not None for param in (
                xsul, xsur, ys, nx, ny
            )])
            if (perform_check and ((xsul - 1) % xbin != 0 or (xsur - 1025) % xbin != 0 or
                                   (ys - 1) % ybin != 0)):
                synced = False

            # Range checks
            rchecks = ((xsll, nx, xsllw), (xslr, nx, xslrw),
                       (xsul, nx, xsulw), (xsur, nx, xsurw),
                       (ys, ny, ysw))
            for check in rchecks:
                val, size, widg = check
                if val is None or size is None or val + size - 1 > widg.imax:
                    widg.config(bg=g.COL['error'])
                    status = False

        # Quad overlap checks. Compare one quad with the next one
        # in the same quadrant if there is one. Only bother if we
        # have survived so far, which saves a lot of checks.
        if status:
            for index in range(nquad-1):
                ys1 = self.ys[index].value()
                ny1 = self.ny[index].value()
                ysw2 = self.ys[index+1]
                ys2 = ysw2.value()
                if any([thing is None for thing in (ys1, ny1, ys2)]) or ys1 + ny1 > ys2:
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