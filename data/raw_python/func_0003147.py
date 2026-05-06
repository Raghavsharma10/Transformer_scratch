def init_scatter_table(self, tm, angular_integration=False, verbose=False):
        """Initialize the scattering lookup tables.
        
        Initialize the scattering lookup tables for the different geometries.
        Before calling this, the following attributes must be set:
           num_points, m_func, axis_ratio_func, D_max, geometries
        and additionally, all the desired attributes of the Scatterer class
        (e.g. wavelength, aspect ratio).

        Args:
            tm: a Scatterer instance.
            angular_integration: If True, also calculate the 
                angle-integrated quantities (scattering cross section, 
                extinction cross section, asymmetry parameter). These are 
                needed to call the corresponding functions in the scatter 
                module when PSD integration is active. The default is False.
            verbose: if True, print information about the progress of the 
                calculation (which may take a while). If False (default), 
                run silently.
        """
        self._psd_D = np.linspace(self.D_max/self.num_points, self.D_max, 
            self.num_points)

        self._S_table = {}
        self._Z_table = {}
        self._previous_psd = None
        self._m_table = np.empty(self.num_points, dtype=complex)
        if angular_integration:
            self._angular_table = {"sca_xsect": {}, "ext_xsect": {}, 
                "asym": {}}
        else:
            self._angular_table = None
        
        (old_m, old_axis_ratio, old_radius, old_geom, old_psd_integrator) = \
            (tm.m, tm.axis_ratio, tm.radius, tm.get_geometry(), 
                tm.psd_integrator)
        
        try:
            # temporarily disable PSD integration to avoid recursion
            tm.psd_integrator = None 

            for geom in self.geometries:
                self._S_table[geom] = \
                    np.empty((2,2,self.num_points), dtype=complex)
                self._Z_table[geom] = np.empty((4,4,self.num_points))

                if angular_integration:
                    for int_var in ["sca_xsect", "ext_xsect", "asym"]:
                        self._angular_table[int_var][geom] = \
                            np.empty(self.num_points)

            for (i,D) in enumerate(self._psd_D):
                if verbose:
                    print("Computing point {i} at D={D}...".format(i=i, D=D))
                if self.m_func != None:
                    tm.m = self.m_func(D)
                if self.axis_ratio_func != None:
                    tm.axis_ratio = self.axis_ratio_func(D)
                self._m_table[i] = tm.m
                tm.radius = D/2.0
                for geom in self.geometries:
                    tm.set_geometry(geom)
                    (S, Z) = tm.get_SZ_orient()
                    self._S_table[geom][:,:,i] = S
                    self._Z_table[geom][:,:,i] = Z

                    if angular_integration:
                        self._angular_table["sca_xsect"][geom][i] = \
                            scatter.sca_xsect(tm)
                        self._angular_table["ext_xsect"][geom][i] = \
                            scatter.ext_xsect(tm)
                        self._angular_table["asym"][geom][i] = \
                            scatter.asym(tm)
        finally:
            #restore old values
            (tm.m, tm.axis_ratio, tm.radius, tm.psd_integrator) = \
                (old_m, old_axis_ratio, old_radius, old_psd_integrator) 
            tm.set_geometry(old_geom)