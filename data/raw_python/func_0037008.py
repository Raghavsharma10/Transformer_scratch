def angular_power_spectrum(self):
        """Returns the angular power spectrum for the set of coefficients.
        That is, we compute

                   n
            c_n = sum  cnm * conj( cnm )
                  m=-n 

        Returns:
          power_spectrum (numpy.array, dtype=double) spectrum as a function of n.
        """

        # Added this routine as a result of my discussions with Ajinkya Nene	 
        #https://github.com/anene
        list_of_modes = self._reshape_m_vecs() 
        Nmodes = len(list_of_modes)

        angular_power = np.zeros( Nmodes, dtype = np.double)

        for n in range(0, Nmodes):
            mode = np.array( list_of_modes[n], dtype = np.complex128 )
            angular_power[n] = np.sum( np.abs(mode) ** 2 )
            
        return angular_power