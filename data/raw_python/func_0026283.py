def get_layers(self, Psurf=1013.25, Ptop=0.01, **kwargs):
        """
        Compute scalars or coordinates associated to the vertical layers.

        Parameters
        ----------
        grid_spec : CTMGrid object
            CTMGrid containing the information necessary to re-construct grid
            levels for a given model coordinate system.

        Returns
        -------
        dictionary of vertical grid components, including eta (unitless),
        sigma (unitless), pressure (hPa), and altitude (km) on both layer centers
        and edges, ordered from bottom-to-top.

        Notes
        -----
        For pure sigma grids, sigma coordinates are given by the esig (edges) and
        csig (centers).

        For both pure sigma and hybrid grids, pressures at layers edges L are
        calculated as follows:

        .. math:: P_e(L) = A_p(L) + B_p(L) * (P_{surf} - C_p)

        where

        :math:`P_{surf}`, :math:`P_{top}`
            Air pressures at the surface and the top of the modeled atmosphere
            (:attr:`Psurf` and :attr:`Ptop` attributes of the :class:`CTMGrid`
            instance).
        :math:`A_p(L)`, :math:`Bp(L)`
            Specified in the grid set-up (`Ap` and `Bp` attributes) for hybrid
            grids, or respectively equals :math:`P_{top}` and :attr:`esig`
            attribute for pure sigma grids.
        :math:`Cp(L)`
            equals :math:`P_{top}` for pure sigma grids or equals 0 for hybrid
            grids.

        Pressures at grid centers are averages of pressures at grid edges:

        .. math:: P_c(L) = (P_e(L) + P_e(L+1)) / 2

        For hybrid grids, ETA coordinates of grid edges and grid centers are
        given by;

        .. math:: ETA_{e}(L) = (P_e(L) - P_{top}) / (P_{surf} - P_{top})
        .. math:: ETA_{c}(L) = (P_c(L) - P_{top}) / (P_{surf} - P_{top})

        Altitude values are fit using a 5th-degree polynomial; see
        `gridspec.prof_altitude` for more details.

        """

        Psurf = np.asarray(Psurf)
        output_ndims = Psurf.ndim + 1
        if output_ndims > 3:
            raise ValueError("`Psurf` argument must be a float or an array"
                             " with <= 2 dimensions (or None)")

        # Compute all variables: takes not much memory, fast
        # and better for code reading
        SIGe = None
        SIGc = None
        ETAe = None
        ETAc = None

        if self.hybrid:
            try:
                Ap = broadcast_1d_array(self.Ap, output_ndims)
                Bp = broadcast_1d_array(self.Bp, output_ndims)
            except KeyError:
                raise ValueError("Impossible to compute vertical levels,"
                                 " data is missing (Ap, Bp)")
            Cp = 0.
        else:
            try:
                Bp = SIGe = broadcast_1d_array(self.esig, output_ndims)
                SIGc = broadcast_1d_array(self.csig, output_ndims)
            except KeyError:
                raise ValueError("Impossible to compute vertical levels,"
                                 " data is missing (esig, csig)")
            Ap = Cp = Ptop

        Pe = Ap + Bp * (Psurf - Cp)
        Pc = 0.5 * (Pe[0:-1] + Pe[1:])

        if self.hybrid:
            ETAe = (Pe - Ptop)/(Psurf - Ptop)
            ETAc = (Pc - Ptop)/(Psurf - Ptop)
        else:
            SIGe = SIGe * np.ones_like(Psurf)
            SIGc = SIGc * np.ones_like(Psurf)

        Ze = prof_altitude(Pe, **kwargs)
        Zc = prof_altitude(Pc, **kwargs)

        all_vars = {'eta_edges': ETAe,
                    'eta_centers': ETAc,
                    'sigma_edges': SIGe,
                    'sigma_centers': SIGc,
                    'pressure_edges': Pe,
                    'pressure_centers': Pc,
                    'altitude_edges': Ze,
                    'altitude_centers': Zc}

        return all_vars