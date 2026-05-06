def set_geoip_params(self, db_country=None, db_city=None):
        """Sets GeoIP parameters.

        * http://uwsgi.readthedocs.io/en/latest/GeoIP.html

        :param str|unicode db_country: Country database file path.

        :param str|unicode db_city: City database file path. Example: ``GeoLiteCity.dat``.

        """
        self._set('geoip-country', db_country, plugin='geoip')
        self._set('geoip-city', db_city, plugin='geoip')

        return self._section