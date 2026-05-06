def sky2ang(sky):
        """
        Convert ra,dec coordinates to theta,phi coordinates
        ra -> phi
        dec -> theta

        Parameters
        ----------
        sky : numpy.array
            Array of (ra,dec) coordinates.
            See :func:`AegeanTools.regions.Region.radec2sky`

        Returns
        -------
        theta_phi : numpy.array
            Array of (theta,phi) coordinates.
        """
        try:
            theta_phi = sky.copy()
        except AttributeError as _:
            theta_phi = np.array(sky)
        theta_phi[:, [1, 0]] = theta_phi[:, [0, 1]]
        theta_phi[:, 0] = np.pi/2 - theta_phi[:, 0]
        # # force 0<=theta<=2pi
        # theta_phi[:, 0] -= 2*np.pi*(theta_phi[:, 0]//(2*np.pi))
        # # and now -pi<=theta<=pi
        # theta_phi[:, 0] -= (theta_phi[:, 0] > np.pi)*2*np.pi
        return theta_phi