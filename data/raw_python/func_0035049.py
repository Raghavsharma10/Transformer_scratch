def _create_waveforms(self):
        """Create the eccentric waveforms

        """

        # find eccentricity and semi major axis over time until e=0.
        e_vals, a_vals, t_vals = self._t_of_e(a0=self.a0, f0=self.f0,
                                              t_start=self.t_start, ef=None,
                                              t_obs=self.t_obs)

        f_mrg = 0.02/(self.m1 + self.m2)
        a_mrg = ((self.m1+self.m2)/f_mrg**2)**(1/3)

        # limit highest frequency to ISCO even though this is not innermost orbit for eccentric
        # binaries
        # find where binary goes farther than observation time or merger frequency limit.
        a_ind_start = np.asarray([np.where(a_vals[i] > a_mrg[i])[0][0] for i in range(len(a_vals))])
        t_ind_start = np.asarray([np.where(t_vals[i] < self.t_obs[i])[0][0]
                                 for i in range(len(t_vals))])

        ind_start = (a_ind_start*(a_ind_start >= t_ind_start)
                     + t_ind_start*(a_ind_start < t_ind_start))

        self.ef = np.asarray([e_vals[i][ind] for i, ind in enumerate(ind_start)])

        # higher resolution over the eccentricities seen during observation
        self.e_vals, self.a_vals, self.t_vals = self._t_of_e(a0=a_vals[:, -1],
                                                             ef=self.ef,
                                                             t_obs=self.t_obs)

        self.freqs_orb = np.sqrt((self.m1[:, np.newaxis]+self.m2[:, np.newaxis])/self.a_vals**3)

        # tile for efficient calculation across modes.
        for attr in ['e_vals', 'a_vals', 't_vals', 'freqs_orb']:
            arr = getattr(self, attr)
            new_arr = (np.flip(
                       np.tile(arr, self.n_max).reshape(len(arr)*self.n_max, len(arr[0])), -1))
            setattr(self, attr, new_arr)

        for attr in ['m1', 'm2', 'z', 'dist']:
            arr = getattr(self, attr)
            new_arr = np.repeat(arr, self.n_max)[:, np.newaxis]
            setattr(self, attr, new_arr)

        # setup modes
        self.n = np.tile(np.arange(1, self.n_max + 1), self.length)[:, np.newaxis]

        self._hcn_func()

        # reshape hc
        self.hc = self.hc.reshape(self.length, self.n_max, self.hc.shape[-1])
        self.freqs = np.reshape(self.n*self.freqs_orb/(1+self.z)
                                * ct.c,
                                (self.length, self.n_max, self.freqs_orb.shape[-1]))

        self.hc, self.freqs = np.squeeze(self.hc), np.squeeze(self.freqs)
        return