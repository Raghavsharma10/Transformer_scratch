def plot(self, title='MOC', frame=None):
        """
        Plot the MOC object using a mollweide projection.

        **Deprecated**: New `fill` and `border` methods produce more reliable results and allow you to specify additional 
        matplotlib style parameters.

        Parameters
        ----------
        title : str
            The title of the plot
        frame : `astropy.coordinates.BaseCoordinateFrame`, optional
            Describes the coordinate system the plot will be (ICRS, Galactic are the only coordinate systems supported).
        """
        frame = ICRS() if frame is None else frame

        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.pyplot as plt

        plot_order = 8
        if self.max_order > plot_order:
            plotted_moc = self.degrade_to_order(plot_order)
        else:
            plotted_moc = self

        num_pixels_map = 1024
        delta = 2. * np.pi / num_pixels_map

        x = np.arange(-np.pi, np.pi, delta)
        y = np.arange(-np.pi/2, np.pi/2, delta)
        lon_rad, lat_rad = np.meshgrid(x, y)
        hp = HEALPix(nside=(1 << plotted_moc.max_order), order='nested')

        if frame and not isinstance(frame, BaseCoordinateFrame):
            raise ValueError("Only Galactic/ICRS coordinate systems are supported."
                             "Please set `coord` to either 'C' or 'G'.")

        pix_map = hp.lonlat_to_healpix(lon_rad * u.rad, lat_rad * u.rad)

        m = np.zeros(nside2npix(1 << plotted_moc.max_order))
        pix_id = plotted_moc._best_res_pixels()

        # change the HEALPix cells if the frame of the MOC is not the same as the one associated with the plot method.
        if isinstance(frame, Galactic):
            lon, lat = hp.boundaries_lonlat(pix_id, step=2)
            sky_crd = SkyCoord(lon, lat, unit='deg')
            pix_id = hp.lonlat_to_healpix(sky_crd.galactic.l, sky_crd.galactic.b)

        m[pix_id] = 1

        z = np.flip(m[pix_map], axis=1)

        plt.figure(figsize=(10, 10))

        ax = plt.subplot(111, projection="mollweide")
        ax.set_xticklabels(['150°', '120°', '90°', '60°', '30°', '0°', '330°', '300°', '270°', '240°', '210°', '180°'])

        color_map = LinearSegmentedColormap.from_list('w2r', ['#eeeeee', '#aa0000'])
        color_map.set_under('w')
        color_map.set_bad('gray')

        ax.pcolormesh(x, y, z, cmap=color_map, vmin=0, vmax=1)
        ax.tick_params(labelsize=14, labelcolor='#000000')
        plt.title(title)
        plt.grid(True, linestyle='--', linewidth=1, color='#555555')

        plt.show()