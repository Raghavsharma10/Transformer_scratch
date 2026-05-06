def from_url(cls, url):
        """
        Creates a `~mocpy.moc.MOC` object from a given url.

        Parameters
        ----------
        url : str
            The url of a FITS file storing a MOC.

        Returns
        -------
        result : `~mocpy.moc.MOC`
            The resulting MOC.
        """
        path = download_file(url, show_progress=False, timeout=60)
        return cls.from_fits(path)