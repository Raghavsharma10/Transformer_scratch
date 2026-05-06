def convert(self, lat, lon, source, dest, height=0, datetime=None,
                precision=1e-10, ssheight=50*6371):
        """Converts between geodetic, modified apex, quasi-dipole and MLT.

        Parameters
        ==========
        lat : array_like
            Latitude
        lon : array_like
            Longitude/MLT
        source : {'geo', 'apex', 'qd', 'mlt'}
            Input coordinate system
        dest : {'geo', 'apex', 'qd', 'mlt'}
            Output coordinate system
        height : array_like, optional
            Altitude in km
        datetime : :class:`datetime.datetime`
            Date and time for MLT conversions (required for MLT conversions)
        precision : float, optional
            Precision of output (degrees) when converting to geo. A negative
            value of this argument produces a low-precision calculation of
            geodetic lat/lon based only on their spherical harmonic
            representation.
            A positive value causes the underlying Fortran routine to iterate
            until feeding the output geo lat/lon into geo2qd (APXG2Q) reproduces
            the input QD lat/lon to within the specified precision (all
            coordinates being converted to geo are converted to QD first and
            passed through APXG2Q).
        ssheight : float, optional
            Altitude in km to use for converting the subsolar point from
            geographic to magnetic coordinates. A high altitude is used
            to ensure the subsolar point is mapped to high latitudes, which
            prevents the South-Atlantic Anomaly (SAA) from influencing the MLT.

        Returns
        =======
        lat : ndarray or float
            Converted latitude (if converting to MLT, output latitude is apex)
        lat : ndarray or float
            Converted longitude/MLT

        """

        if datetime is None and ('mlt' in [source, dest]):
            raise ValueError('datetime must be given for MLT calculations')

        lat = helpers.checklat(lat)

        if source == dest:
            return lat, lon
        # from geo
        elif source == 'geo' and dest == 'apex':
            lat, lon = self.geo2apex(lat, lon, height)
        elif source == 'geo' and dest == 'qd':
            lat, lon = self.geo2qd(lat, lon, height)
        elif source == 'geo' and dest == 'mlt':
            lat, lon = self.geo2apex(lat, lon, height)
            lon = self.mlon2mlt(lon, datetime, ssheight=ssheight)
        # from apex
        elif source == 'apex' and dest == 'geo':
            lat, lon, _ = self.apex2geo(lat, lon, height, precision=precision)
        elif source == 'apex' and dest == 'qd':
            lat, lon = self.apex2qd(lat, lon, height=height)
        elif source == 'apex' and dest == 'mlt':
            lon = self.mlon2mlt(lon, datetime, ssheight=ssheight)
        # from qd
        elif source == 'qd' and dest == 'geo':
            lat, lon, _ = self.qd2geo(lat, lon, height, precision=precision)
        elif source == 'qd' and dest == 'apex':
            lat, lon = self.qd2apex(lat, lon, height=height)
        elif source == 'qd' and dest == 'mlt':
            lat, lon = self.qd2apex(lat, lon, height=height)
            lon = self.mlon2mlt(lon, datetime, ssheight=ssheight)
        # from mlt (input latitude assumed apex)
        elif source == 'mlt' and dest == 'geo':
            lon = self.mlt2mlon(lon, datetime, ssheight=ssheight)
            lat, lon, _ = self.apex2geo(lat, lon, height, precision=precision)
        elif source == 'mlt' and dest == 'apex':
            lon = self.mlt2mlon(lon, datetime, ssheight=ssheight)
        elif source == 'mlt' and dest == 'qd':
            lon = self.mlt2mlon(lon, datetime, ssheight=ssheight)
            lat, lon = self.apex2qd(lat, lon, height=height)
        # no other transformations are implemented
        else:
            estr = 'Unknown coordinate transformation: '
            estr += '{} -> {}'.format(source, dest)
            raise NotImplementedError(estr)

        return lat, lon