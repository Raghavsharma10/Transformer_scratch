def catalog(self):
        """Create MOC from catalog of coordinates.

        This command requires that the Healpy and Astropy libraries
        be available.  It attempts to load the given catalog,
        and merges it with the running MOC.

        The name of an ASCII catalog file should be given.  The file
        should contain either "RA" and "Dec" columns (for ICRS coordinates)
        or "Lon" and "Lat" columns (for galactic coordinates).  The MOC
        order and radius (in arcseconds) can be given with additional
        options.

        ::

            pymoctool --catalog coords.txt
                [order 12]
                [radius 3600]
                [unit (hour | deg | rad) (deg | rad)]
                [format commented_header]
                [inclusive]

        Units (if not specified) are assumed to be hours and degrees for ICRS
        coordinates and degrees for galactic coordinates.  The format, if not
        specified (as an Astropy ASCII table format name) is assumed to be
        commented header, e.g.:

        ::

            # RA Dec
            01:30:00 +45:00:00
            22:30:00 +45:00:00
        """

        from .catalog import catalog_to_moc, read_ascii_catalog

        filename = self.params.pop()
        order = 12
        radius = 3600
        unit = None
        format_ = 'commented_header'
        kwargs = {}

        while self.params:
            if self.params[-1] == 'order':
                self.params.pop()
                order = int(self.params.pop())
            elif self.params[-1] == 'radius':
                self.params.pop()
                radius = float(self.params.pop())
            elif self.params[-1] == 'unit':
                self.params.pop()
                unit_x = self.params.pop()
                unit_y = self.params.pop()
                unit = (unit_x, unit_y)
            elif self.params[-1] == 'format':
                self.params.pop()
                format_ = self.params.pop()
            elif self.params[-1] == 'inclusive':
                self.params.pop()
                kwargs['inclusive'] = True
            else:
                break

        coords = read_ascii_catalog(filename, format_=format_, unit=unit)
        catalog_moc = catalog_to_moc(coords, radius, order, **kwargs)

        if self.moc is None:
            self.moc = catalog_moc
        else:
            self.moc += catalog_moc