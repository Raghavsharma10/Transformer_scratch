def get_period_LS(self, date, mag, n_threads, min_period):
        """
        Period finding using the Lomb-Scargle algorithm.

        Finding two periods. The second period is estimated after whitening
        the first period. Calculating various other features as well
        using derived periods.

        Parameters
        ----------
        date : array_like
            An array of observed date, in days.
        mag : array_like
            An array of observed magnitude.
        n_threads : int
            The number of threads to use.
        min_period : float
            The minimum period to calculate.
        """

        # DO NOT CHANGE THESE PARAMETERS.
        oversampling = 3.
        hifac = int((max(date) - min(date)) / len(date) / min_period * 2.)

        # Minimum hifac
        if hifac < 100:
            hifac = 100

        # Lomb-Scargle.
        fx, fy, nout, jmax, prob = pLS.fasper(date, mag, oversampling, hifac,
                                              n_threads)

        self.f = fx[jmax]
        self.period = 1. / self.f
        self.period_uncertainty = self.get_period_uncertainty(fx, fy, jmax)
        self.period_log10FAP = \
            np.log10(pLS.getSignificance(fx, fy, nout, oversampling)[jmax])
        # self.f_SNR1 = fy[jmax] / np.median(fy)
        self.period_SNR = (fy[jmax] - np.median(fy)) / np.std(fy)

        # Fit Fourier Series of order 3.
        order = 3
        # Initial guess of Fourier coefficients.
        p0 = np.ones(order * 2 + 1)
        date_period = (date % self.period) / self.period
        p1, success = leastsq(self.residuals, p0,
                              args=(date_period, mag, order))
        # fitted_y = self.FourierSeries(p1, date_period, order)

        # print p1, self.mean, self.median
        # plt.plot(date_period, self.mag, 'b+')
        # plt.show()

        # Derive Fourier features for the first period.
        # Petersen, J. O., 1986, A&A
        self.amplitude = np.sqrt(p1[1] ** 2 + p1[2] ** 2)
        self.r21 = np.sqrt(p1[3] ** 2 + p1[4] ** 2) / self.amplitude
        self.r31 = np.sqrt(p1[5] ** 2 + p1[6] ** 2) / self.amplitude
        self.f_phase = np.arctan(-p1[1] / p1[2])
        self.phi21 = np.arctan(-p1[3] / p1[4]) - 2. * self.f_phase
        self.phi31 = np.arctan(-p1[5] / p1[6]) - 3. * self.f_phase

        """
        # Derive a second period.
        # Whitening a light curve.
        residual_mag = mag - fitted_y

        # Lomb-Scargle again to find the second period.
        omega_top, power_top = search_frequencies(date, residual_mag, err,
            #LS_kwargs={'generalized':True, 'subtract_mean':True},
            n_eval=5000, n_retry=3, n_save=50)

        self.period2 = 2*np.pi/omega_top[np.where(power_top==np.max(power_top))][0]
        self.f2 = 1. / self.period2
        self.f2_SNR = power_top[np.where(power_top==np.max(power_top))][0] \
            * (len(self.date) - 1) / 2.

        # Fit Fourier Series again.
        p0 = [1.] * order * 2
        date_period = (date % self.period) / self.period
        p2, success = leastsq(self.residuals, p0,
            args=(date_period, residual_mag, order))
        fitted_y = self.FourierSeries(p2, date_period, order)

        #plt.plot(date%self.period2, residual_mag, 'b+')
        #plt.show()

        # Derive Fourier features for the first second.
        self.f2_amp = 2. * np.sqrt(p2[1]**2 + p2[2]**2)
        self.f2_R21 = np.sqrt(p2[3]**2 + p2[4]**2) / self.f2_amp
        self.f2_R31 = np.sqrt(p2[5]**2 + p2[6]**2) / self.f2_amp
        self.f2_R41 = np.sqrt(p2[7]**2 + p2[8]**2) / self.f2_amp
        self.f2_R51 = np.sqrt(p2[9]**2 + p2[10]**2) / self.f2_amp
        self.f2_phase = np.arctan(-p2[1] / p2[2])
        self.f2_phi21 = np.arctan(-p2[3] / p2[4]) - 2. * self.f2_phase
        self.f2_phi31 = np.arctan(-p2[5] / p2[6]) - 3. * self.f2_phase
        self.f2_phi41 = np.arctan(-p2[7] / p2[8]) - 4. * self.f2_phase
        self.f2_phi51 = np.arctan(-p2[9] / p2[10]) - 5. * self.f2_phase

        # Calculate features using the first and second periods.
        self.f12_ratio = self.f2 / self.f1
        self.f12_remain = self.f1 % self.f2 \
            if self.f1 > self.f2 else self.f2 % self.f1
        self.f12_amp = self.f2_amp / self.f1_amp
        self.f12_phase = self.f2_phase - self.f1_phase
        """