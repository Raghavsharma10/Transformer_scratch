def timing(self):
        """
        Estimates timing information for the current setup. You should
        run a check on the instrument parameters before calling this.

        Returns: (expTime, deadTime, cycleTime, dutyCycle)

        expTime   : exposure time per frame (seconds)
        deadTime  : dead time per frame (seconds)
        cycleTime : sampling time (cadence), (seconds)
        dutyCycle : percentage time exposing.
        frameRate : number of frames per second
        """
        # drift mode y/n?
        isDriftMode = self.isDrift()
        # FF y/n?
        isFF = self.isFF()

        # Set the readout speed
        readSpeed = self.readSpeed()

        if readSpeed == 'Fast' and self.dummy():
            video = VIDEO_FAST
        elif readSpeed == 'Slow' and self.dummy():
            video = VIDEO_SLOW
        elif not self.dummy():
            video = VIDEO_SLOW_SE
        else:
            raise DriverError('InstPars.timing: readout speed = ' +
                              readSpeed + ' not recognised.')

        if self.fastClk():
            DUMP_TIME = DUMP_TIME_FAST
            VCLOCK_FRAME = VCLOCK_FAST
            VCLOCK_STORAGE = VCLOCK_FAST
            HCLOCK = HCLOCK_FAST
        else:
            DUMP_TIME = DUMP_TIME_SLOW
            VCLOCK_FRAME = VCLOCK_FRAME_SLOW
            VCLOCK_STORAGE = VCLOCK_STORAGE_SLOW
            HCLOCK = HCLOCK_SLOW

        # clear chip on/off?
        lclear = not isDriftMode and self.clear()

        # overscan read or not
        oscan = not isDriftMode and self.oscan()
        oscany = not isDriftMode and self.oscany()

        # get exposure delay
        expose = self.expose.value()

        # window parameters
        xbin = self.wframe.xbin.value()
        ybin = self.wframe.ybin.value()
        if isDriftMode:
            nwin = 1  # number of windows per output
            dys = self.wframe.ys[0].value() - 1
            dnx = self.wframe.nx[0].value()
            dny = self.wframe.ny[0].value()
            dxsl = self.wframe.xsl[0].value()
            dxsr = self.wframe.xsr[0].value()
            # differential shift needed to line both
            # windows up with the edge of the chip
            diffshift = abs(dxsl - 1 - (2*FFX - dxsr - dnx + 1))
        elif isFF:
            nwin = 1
            ys, nx, ny = [0], [1024], [512]
        else:
            ys, nx, ny = [], [], []
            xse, xsf, xsg, xsh = [], [], [], []
            nwin = self.wframe.nquad.value()
            for xsll, xsul, xslr, xsur, ysv, nxv, nyv in self.wframe:
                xse.append(xsll - 1)
                xsf.append(2049 - xslr - nxv)
                xsg.append(2049 - xsur - nxv)
                xsh.append(xsul - 1)
                ys.append(ysv-1)
                nx.append(nxv)
                ny.append(nyv)

        # convert timing parameters to seconds
        expose_delay = expose

        # clear chip by VCLOCK-ing the image and area and dumping storage area (x5)
        if lclear:
            clear_time = 5*(FFY*VCLOCK_FRAME + FFY*DUMP_TIME)
        else:
            clear_time = 0.0

        if isDriftMode:
            # for drift mode, we need the number of windows in the pipeline
            # and the pipeshift
            nrows = FFY  # number of rows in storage area
            pnwin = int(((nrows / dny) + 1)/2)
            pshift = nrows - (2*pnwin-1)*dny
            frame_transfer = (dny+dys)*VCLOCK_FRAME

            yshift = [dys*VCLOCK_STORAGE]

            # After placing the window adjacent to the serial register, the
            # register must be cleared by clocking out the entire register,
            # taking FFX hclocks.
            line_clear = [0.]
            if yshift[0] != 0:
                line_clear[0] = DUMP_TIME

            # to calculate number of HCLOCKS needed to read a line in
            # drift mode we have to account for the diff shifts and dumping.
            # first perform diff shifts
            # for now we need this *2 (for quadrants E, H or F, G)
            numhclocks = 2*diffshift
            # now add the amount of clocks needed to get
            # both windows to edge of chip
            if dxsl - 1 > 2*FFX - dxsr - dnx + 1:
                # it was the left window that got the diff shift,
                # so the number of hclocks increases by the amount
                # needed to get the RH window to the edge
                numhclocks += 2*FFX - dxsr - dnx + 1
            else:
                # vice versa
                numhclocks += dxsl - 1
            # now we actually clock the windows themselves
            numhclocks += dnx
            # finally, we need to hclock the additional pre-scan pixels
            numhclocks += 2*PRSCX

            # here is the total time to read the whole line
            line_read = [VCLOCK_STORAGE*ybin + numhclocks*HCLOCK +
                         video*dnx/xbin + DUMP_TIME + 2*SETUP_READ]

            readout = [(dny/ybin) * line_read[0]]
        elif isFF:
            # move entire image into storage area
            frame_transfer = FFY*VCLOCK_FRAME + DUMP_TIME

            yshift = [0]
            line_clear = [0]

            numhclocks = FFX + PRSCX
            line_read = [VCLOCK_STORAGE*ybin + numhclocks*HCLOCK +
                         video*nx[0]/xbin + SETUP_READ]
            if oscan:
                line_read[0] += video*PRSCX/xbin
            nlines = ny[0]/ybin if not oscany else (ny[0] + 8/ybin)
            readout = [nlines*line_read[0]]
        else:
            # windowed mode
            # move entire image into storage area
            frame_transfer = FFY*VCLOCK_FRAME + DUMP_TIME

            # dump rows in storage area up to start of the window without changing the
            # image area.
            yshift = nwin*[0.]
            yshift[0] = ys[0]*DUMP_TIME
            for nw in range(1, nwin):
                yshift[nw] = (ys[nw]-ys[nw-1]-ny[nw-1])*DUMP_TIME

            line_clear = nwin*[0.]
            # Naidu always dumps the serial register, in windowed mode
            # regardless of whether we need to or not
            for nw in range(nwin):
                line_clear[nw] = DUMP_TIME

            # calculate how long it takes to shift one row into the serial
            # register shift along serial register and then read out the data.
            # total number of hclocks needs to account for diff shifts of
            # windows, carried out in serial
            numhclocks = nwin*[0]
            for nw in range(nwin):
                common_shift = min(xse[nw], xsf[nw], xsg[nw], xsh[nw])
                diffshifts = sum((xs-common_shift for xs in (xse[nw], xsf[nw], xsg[nw], xsh[nw])))
                numhclocks[nw] = 2*PRSCX + common_shift + diffshifts + nx[nw]

            line_read = nwin*[0.]
            # line read includes vclocking a row, all the hclocks, digitising pixels and dumping serial register
            # when windows are read out.
            for nw in range(nwin):
                line_read[nw] = (VCLOCK_STORAGE*ybin + numhclocks[nw]*HCLOCK +
                                 video*nx[nw]/xbin + 2*SETUP_READ + DUMP_TIME)
                if oscan:
                    line_read[nw] += video*PRSCX/xbin

            # multiply time to shift one row into serial register by
            # number of rows for total readout time
            readout = nwin*[0.]
            for nw in range(nwin):
                nlines = ny[nw]/ybin if not oscany else (ny[nw] + 8/ybin)
                readout[nw] = nlines * line_read[nw]

        # now get the total time to read out one exposure.
        cycleTime = expose_delay + clear_time + frame_transfer
        if isDriftMode:
            cycleTime += pshift*VCLOCK_STORAGE + yshift[0] + line_clear[0] + readout[0]
        else:
            for nw in range(nwin):
                cycleTime += yshift[nw] + line_clear[nw] + readout[nw]

        # use 5sec estimate for nod time
        # TODO: replace with accurate estimate
        if self.nod() and lclear:
            cycleTime += 5
        elif self.nod():
            g = get_root(self).globals
            g.clog.warn('ERR: dithering enabled with clear mode off')

        frameRate = 1.0/cycleTime
        expTime = expose_delay if lclear else cycleTime - frame_transfer
        deadTime = cycleTime - expTime
        dutyCycle = 100.0*expTime/cycleTime
        return (expTime, deadTime, cycleTime, dutyCycle, frameRate)