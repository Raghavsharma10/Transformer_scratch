def update(self, *args):
        """
        Updates values. You should run a check on the instrument and
        target parameters before calling this.
        """
        g = get_root(self).globals
        expTime, deadTime, cycleTime, dutyCycle, frameRate = g.ipars.timing()

        total, peak, peakSat, peakWarn, ston, ston3 = \
            self.counts(expTime, cycleTime)

        if cycleTime < 0.01:
            self.cadence.config(text='{0:7.5f} s'.format(cycleTime))
        elif cycleTime < 0.1:
            self.cadence.config(text='{0:6.4f} s'.format(cycleTime))
        elif cycleTime < 1.:
            self.cadence.config(text='{0:5.3f} s'.format(cycleTime))
        elif cycleTime < 10.:
            self.cadence.config(text='{0:4.2f} s'.format(cycleTime))
        elif cycleTime < 100.:
            self.cadence.config(text='{0:4.1f} s'.format(cycleTime))
        elif cycleTime < 1000.:
            self.cadence.config(text='{0:4.0f} s'.format(cycleTime))
        else:
            self.cadence.config(text='{0:5.0f} s'.format(cycleTime))

        if expTime < 0.01:
            self.exposure.config(text='{0:7.5f} s'.format(expTime))
        elif expTime < 0.1:
            self.exposure.config(text='{0:6.4f} s'.format(expTime))
        elif expTime < 1.:
            self.exposure.config(text='{0:5.3f} s'.format(expTime))
        elif expTime < 10.:
            self.exposure.config(text='{0:4.2f} s'.format(expTime))
        elif expTime < 100.:
            self.exposure.config(text='{0:4.1f} s'.format(expTime))
        elif expTime < 1000.:
            self.exposure.config(text='{0:4.0f} s'.format(expTime))
        else:
            self.exposure.config(text='{0:5.0f} s'.format(expTime))

        self.duty.config(text='{0:4.1f} %'.format(dutyCycle))
        self.peak.config(text='{0:d} cts'.format(int(round(peak))))
        if peakSat:
            self.peak.config(bg=g.COL['error'])
        elif peakWarn:
            self.peak.config(bg=g.COL['warn'])
        else:
            self.peak.config(bg=g.COL['main'])

        self.total.config(text='{0:d} cts'.format(int(round(total))))
        self.ston.config(text='{0:.1f}'.format(ston))
        self.ston3.config(text='{0:.1f}'.format(ston3))