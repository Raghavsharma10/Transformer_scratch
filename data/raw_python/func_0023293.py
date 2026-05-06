def fill(self, ax, wcs, **kw_mpl_pathpatch):
        """
        Draws the MOC on a matplotlib axis.

        This performs the projection of the cells from the world coordinate system to the pixel image coordinate system.
        You are able to specify various styling kwargs for `matplotlib.patches.PathPatch`
        (see the `list of valid keywords <https://matplotlib.org/api/_as_gen/matplotlib.patches.PathPatch.html#matplotlib.patches.PathPatch>`__).

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Matplotlib axis.
        wcs : `astropy.wcs.WCS`
            WCS defining the World system <-> Image system projection.
        kw_mpl_pathpatch
            Plotting arguments for `matplotlib.patches.PathPatch`.
        
        Examples
        --------
        >>> from mocpy import MOC, WCS
        >>> from astropy.coordinates import Angle, SkyCoord
        >>> import astropy.units as u
        >>> # Load a MOC, e.g. the MOC of GALEXGR6-AIS-FUV
        >>> filename = './../resources/P-GALEXGR6-AIS-FUV.fits'
        >>> moc = MOC.from_fits(filename)
        >>> # Plot the MOC using matplotlib
        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure(111, figsize=(15, 15))
        >>> # Define a WCS as a context
        >>> with WCS(fig, 
        ...         fov=50 * u.deg,
        ...         center=SkyCoord(0, 20, unit='deg', frame='icrs'),
        ...         coordsys="icrs",
        ...         rotation=Angle(0, u.degree),
        ...         projection="AIT") as wcs:
        ...     ax = fig.add_subplot(1, 1, 1, projection=wcs)
        ...     # Call fill giving the matplotlib axe and the `~astropy.wcs.WCS` object.
        ...     # We will set the matplotlib keyword linewidth to 0 so that it does not plot
        ...     # the border of each HEALPix cell.
        ...     # The color can also be specified along with an alpha value.
        ...     moc.fill(ax=ax, wcs=wcs, linewidth=0, alpha=0.5, fill=True, color="green")
        >>> plt.xlabel('ra')
        >>> plt.ylabel('dec')
        >>> plt.grid(color="black", linestyle="dotted")
        """
        fill.fill(self, ax, wcs, **kw_mpl_pathpatch)