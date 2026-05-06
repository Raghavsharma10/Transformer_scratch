def counts(self, expTime, cycleTime, ap_scale=1.6, ndiv=5):
        """
        Computes counts per pixel, total counts, sky counts
        etc given current magnitude, seeing etc. You should
        run a check on the instrument parameters before calling
        this.

        expTime   : exposure time per frame (seconds)
        cycleTime : sampling, cadence (seconds)
        ap_scale  : aperture radius as multiple of seeing

        Returns: (total, peak, peakSat, peakWarn, ston, ston3)

        total    -- total number of object counts in aperture
        peak     -- peak counts in a pixel
        peakSat  -- flag to indicate saturation
        peakWarn -- flag to indication level approaching saturation
        ston     -- signal-to-noise per exposure
        ston3    -- signal-to-noise after 3 hours on target
        """

        # code directly translated from Java equivalent.
        g = get_root(self).globals

        # Set the readout speed
        readSpeed = g.ipars.readSpeed()
        if readSpeed == 'Fast':
            gain = GAIN_FAST
            read = RNO_FAST
        elif readSpeed == 'Slow':
            gain = GAIN_SLOW
            read = RNO_SLOW
        else:
            raise DriverError('CountsFrame.counts: readout speed = ' +
                              readSpeed + ' not recognised.')

        xbin, ybin = g.ipars.wframe.xbin.value(), g.ipars.wframe.ybin.value()

        # calculate SN info.
        zero, sky, skyTot, darkTot = 0., 0., 0., 0.
        total, peak, correct, signal, readTot, seeing = 0., 0., 0., 0., 0., 0.
        noise, narcsec, npix, signalToNoise3 = 1., 0., 0., 0.

        tinfo = g.TINS[g.cpars['telins_name']]
        filtnam = self.filter.value()

        zero = tinfo['zerop'][filtnam]
        mag = self.mag.value()
        seeing = self.seeing.value()
        sky = g.SKY[self.moon.value()][filtnam]
        airmass = self.airmass.value()
        plateScale = tinfo['plateScale']

        # calculate expected electrons
        total = 10.**((zero-mag-airmass*g.EXTINCTION[filtnam])/2.5)*expTime

        # compute fraction that fall in central pixel
        # assuming target exactly at its centre. Do this
        # by splitting each pixel of the central (potentially
        # binned) pixel into ndiv * ndiv points at
        # which the seeing profile is added. sigma is the
        # RMS seeing in terms of pixels.
        sigma = seeing/g.EFAC/plateScale

        sum = 0.
        for iyp in range(ybin):
            yoff = -ybin/2.+iyp
            for ixp in range(xbin):
                xoff = -xbin/2.+ixp
                for iys in range(ndiv):
                    y = (yoff + (iys+0.5)/ndiv)/sigma
                    for ixs in range(ndiv):
                        x = (xoff + (ixs+0.5)/ndiv)/sigma
                        sum += math.exp(-(x*x+y*y)/2.)
        peak = total*sum/(2.*math.pi*sigma**2*ndiv**2)

        # Work out fraction of flux in aperture with radius AP_SCALE*seeing
        correct = 1. - math.exp(-(g.EFAC*ap_scale)**2/2.)

        # expected sky e- per arcsec
        skyPerArcsec = 10.**((zero-sky)/2.5)*expTime
        # skyPerPixel = skyPerArcsec*plateScale**2*xbin*ybin
        narcsec = math.pi*(ap_scale*seeing)**2
        skyTot = skyPerArcsec*narcsec
        npix = math.pi*(ap_scale*seeing/plateScale)**2/xbin/ybin

        signal = correct*total  # in electrons
        darkTot = npix*DARK_E*expTime  # in electrons
        readTot = npix*read**2  # in electrons

        # noise, in electrons
        noise = math.sqrt(readTot + darkTot + skyTot + signal)

        # Now compute signal-to-noise in 3 hour seconds run
        signalToNoise3 = signal/noise*math.sqrt(3*3600./cycleTime)

        # convert from electrons to counts
        total /= gain
        peak /= gain

        warn = 25000
        sat = 60000

        peakSat = peak > sat
        peakWarn = peak > warn

        return (total, peak, peakSat, peakWarn, signal/noise, signalToNoise3)