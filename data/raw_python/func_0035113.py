def set_parameters(self):
        """Setup all the parameters for the binaries to be evaluated.

        Grid values and store necessary parameters for input into the SNR function.

        """

        # declare 1D arrays of both paramters
        if self.xscale != 'lin':
            self.xvals = np.logspace(np.log10(float(self.x_low)),
                                     np.log10(float(self.x_high)),
                                     self.num_x)

        else:
            self.xvals = np.linspace(float(self.x_low),
                                     float(self.x_high),
                                     self.num_x)

        if self.yscale != 'lin':
            self.yvals = np.logspace(np.log10(float(self.y_low)),
                                     np.log10(float(self.y_high)),
                                     self.num_y)

        else:
            self.yvals = np.linspace(float(self.y_low),
                                     float(self.y_high),
                                     self.num_y)

        self.xvals, self.yvals = np.meshgrid(self.xvals, self.yvals)
        self.xvals, self.yvals = self.xvals.ravel(), self.yvals.ravel()

        for which in ['x', 'y']:
            setattr(self, getattr(self, which + 'val_name'), getattr(self, which + 'vals'))

        self.ecc = 'eccentricity' in self.__dict__
        if self.ecc:
            if 'observation_time' not in self.__dict__:
                if 'start_time' not in self.__dict__:
                    raise ValueError('If no observation time is provided, the time before'
                                     + 'merger must be the inital starting condition.')
                self.observation_time = self.start_time  # small number so it is not zero
        else:
            if 'spin' in self.__dict__:
                self.spin_1 = self.spin
                self.spin_2 = self.spin

        for key in ['redshift', 'luminosity_distance', 'comoving_distance']:
            if key in self.__dict__:
                self.dist_type = key
                self.z_or_dist = getattr(self, key)

            if self.ecc:
                for key in ['start_frequency', 'start_time', 'start_separation']:
                    if key in self.__dict__:
                        self.initial_cond_type = key.split('_')[-1]
                        self.initial_point = getattr(self, key)

        # add m1 and m2
        self.m1 = (self.total_mass / (1. + self.mass_ratio))
        self.m2 = (self.total_mass * self.mass_ratio / (1. + self.mass_ratio))
        return