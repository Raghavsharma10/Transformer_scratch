def write_fits(self, filename, moctool=''):
        """
        Write a fits file representing the MOC of this region.

        Parameters
        ----------
        filename : str
            File to write

        moctool : str
            String to be written to fits header with key "MOCTOOL".
            Default = ''
        """
        datafile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'MOC.fits')
        hdulist = fits.open(datafile)
        cols = fits.Column(name='NPIX', array=self._uniq(), format='1K')
        tbhdu = fits.BinTableHDU.from_columns([cols])
        hdulist[1] = tbhdu
        hdulist[1].header['PIXTYPE'] = ('HEALPIX ', 'HEALPix magic code')
        hdulist[1].header['ORDERING'] = ('NUNIQ ', 'NUNIQ coding method')
        hdulist[1].header['COORDSYS'] = ('C ', 'ICRS reference frame')
        hdulist[1].header['MOCORDER'] = (self.maxdepth, 'MOC resolution (best order)')
        hdulist[1].header['MOCTOOL'] = (moctool, 'Name of the MOC generator')
        hdulist[1].header['MOCTYPE'] = ('CATALOG', 'Source type (IMAGE or CATALOG)')
        hdulist[1].header['MOCID'] = (' ', 'Identifier of the collection')
        hdulist[1].header['ORIGIN'] = (' ', 'MOC origin')
        time = datetime.datetime.utcnow()
        hdulist[1].header['DATE'] = (datetime.datetime.strftime(time, format="%Y-%m-%dT%H:%m:%SZ"), 'MOC creation date')
        hdulist.writeto(filename, overwrite=True)
        return