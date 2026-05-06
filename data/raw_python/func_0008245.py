def loadJSON(self, json_string):
        """
        Loads in an application saved in JSON format.
        """
        g = get_root(self).globals
        data = json.loads(json_string)['appdata']
        # first set the parameters which change regardless of mode
        # number of exposures
        numexp = data.get('numexp', 0)
        if numexp == -1:
            numexp = 0
        self.number.set(numexp)
        # Overscan (x, y)
        if 'oscan' in data:
            self.oscan.set(data['oscan'])
        if 'oscany' in data:
            self.oscan.set(data['oscany'])
        # LED setting
        self.led.set(data.get('led_flsh', 0))
        # Dummy output enabled
        self.dummy.set(data.get('dummy_out', 0))
        # Fast clocking option?
        self.fastClk.set(data.get('fast_clks', 0))
        # readout speed
        self.readSpeed.set(data.get('readout', 'Slow'))
        # dwell
        dwell = data.get('dwell', 0)
        self.expose.set(str(float(dwell)))

        # multipliers
        mult_values = data.get('multipliers',
                               (1, 1, 1, 1, 1))
        self.nmult.setall(mult_values)

        # look for nodpattern in data
        nodPattern = data.get('nodpattern', {})
        if nodPattern and g.cpars['telins_name'] == 'GTC':
            self.nodPattern = nodPattern
            self.nod.set(True)
            self.clear.set(True)
        else:
            self.nodPattern = {}
            self.nod.set(False)

        # binning
        self.quad_frame.xbin.set(data.get('xbin', 1))
        self.quad_frame.ybin.set(data.get('ybin', 1))
        self.drift_frame.xbin.set(data.get('xbin', 1))
        self.drift_frame.ybin.set(data.get('ybin', 1))

        # now for the behaviour which depends on mode
        if 'app' in data:
            self.app.set(data['app'])
            app = data['app']

            if app == 'Drift':
                # disable clear mode in drift
                self.clear.set(0)
                # only one pair allowed
                self.wframe.npair.set(1)

                # set the window pair values
                labels = ('x1start_left', 'y1start',
                          'x1start_right', 'x1size',
                          'y1size')
                if not all(label in data for label in labels):
                    raise DriverError('Drift mode application missing window params')
                # now actually set them
                self.wframe.xsl[0].set(data['x1start_left'])
                self.wframe.xsr[0].set(data['x1start_right'])
                self.wframe.ys[0].set(data['y1start'])
                self.wframe.nx[0].set(data['x1size'])
                self.wframe.ny[0].set(data['y1size'])
                self.wframe.check()

            elif app == 'FullFrame':
                # enable clear mode if set
                self.clear.set(data.get('clear', 0))

            elif app == 'Windows':
                # enable clear mode if set
                self.clear.set(data.get('clear', 0))
                nquad = 0
                for nw in range(2):
                    labels = ('x{0}start_lowerleft y{0}start x{0}start_upperleft x{0}start_upperright ' +
                              'x{0}start_lowerright x{0}size y{0}size').format(nw+1).split()
                    if all(label in data for label in labels):
                        xsll = data[labels[0]]
                        xslr = data[labels[4]]
                        xsul = data[labels[2]]
                        xsur = data[labels[3]]
                        ys = data[labels[1]]
                        nx = data[labels[5]]
                        ny = data[labels[6]]
                        self.wframe.xsll[nw].set(xsll)
                        self.wframe.xslr[nw].set(xslr)
                        self.wframe.xsul[nw].set(xsul)
                        self.wframe.xsur[nw].set(xsur)
                        self.wframe.ys[nw].set(ys)
                        self.wframe.nx[nw].set(nx)
                        self.wframe.ny[nw].set(ny)
                        nquad += 1
                    else:
                        break
                self.wframe.nquad.set(nquad)
                self.wframe.check()