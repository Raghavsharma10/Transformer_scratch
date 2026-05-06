def dumpJSON(self):
        """
        Encodes current parameters to JSON compatible dictionary
        """
        numexp = self.number.get()
        expTime, _, _, _, _ = self.timing()
        if numexp == 0:
            numexp = -1

        data = dict(
            numexp=self.number.value(),
            app=self.app.value(),
            led_flsh=self.led(),
            dummy_out=self.dummy(),
            fast_clks=self.fastClk(),
            readout=self.readSpeed(),
            dwell=self.expose.value(),
            exptime=expTime,
            oscan=self.oscan(),
            oscany=self.oscany(),
            xbin=self.wframe.xbin.value(),
            ybin=self.wframe.ybin.value(),
            multipliers=self.nmult.getall(),
            clear=self.clear()
        )

        # only allow nodding in clear mode, even if GUI has got confused
        if data['clear'] and self.nodPattern:
            data['nodpattern'] = self.nodPattern

        # no mixing clear and multipliers, no matter what GUI says
        if data['clear']:
            data['multipliers'] = [1 for i in self.nmult.getall()]

        # add window mode
        if not self.isFF():
            if self.isDrift():
                # no clear, multipliers or oscan in drift
                for setting in ('clear', 'oscan', 'oscany'):
                    data[setting] = 0
                data['multipliers'] = [1 for i in self.nmult.getall()]

                for iw, (xsl, xsr, ys, nx, ny) in enumerate(self.wframe):
                    data['x{}start_left'.format(iw+1)] = xsl
                    data['x{}start_right'.format(iw+1)] = xsr
                    data['y{}start'.format(iw+1)] = ys
                    data['y{}size'.format(iw+1)] = ny
                    data['x{}size'.format(iw+1)] = nx
            else:
                # no oscany in window mode
                data['oscany'] = 0

                for iw, (xsll, xsul, xslr, xsur, ys, nx, ny) in enumerate(self.wframe):
                    data['x{}start_upperleft'.format(iw+1)] = xsul
                    data['x{}start_lowerleft'.format(iw+1)] = xsll
                    data['x{}start_upperright'.format(iw+1)] = xsur
                    data['x{}start_lowerright'.format(iw+1)] = xslr
                    data['y{}start'.format(iw+1)] = ys
                    data['x{}size'.format(iw+1)] = nx
                    data['y{}size'.format(iw+1)] = ny
        return data