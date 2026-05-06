def from_fits_images(cls, path_l, max_norder):
        """
        Loads a MOC from a set of FITS file images.

        Parameters
        ----------
        path_l : [str]
            A list of path where the fits image are located.
        max_norder : int
            The MOC resolution.

        Returns
        -------
        moc : `~mocpy.moc.MOC`
            The union of all the MOCs created from the paths found in ``path_l``.
        """
        moc = MOC()
        for path in path_l:
            header = fits.getheader(path)
            current_moc = MOC.from_image(header=header, max_norder=max_norder)
            moc = moc.union(current_moc)

        return moc