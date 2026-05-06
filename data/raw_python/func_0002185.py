def get_catalog(self, query):
        """Fetch a parsed THREDDS catalog from the radar server.

        Requests a catalog of radar data files data from the radar server given the
        parameters in `query` and returns a :class:`~siphon.catalog.TDSCatalog` instance.

        Parameters
        ----------
        query : RadarQuery
            The parameters to send to the radar server

        Returns
        -------
        catalog : TDSCatalog
            The catalog of matching data files

        Raises
        ------
        :class:`~siphon.http_util.BadQueryError`
            When the query cannot be handled by the server

        See Also
        --------
        get_catalog_raw

        """
        # TODO: Refactor TDSCatalog so we don't need two requests, or to do URL munging
        try:
            url = self._base[:-1] if self._base[-1] == '/' else self._base
            url += '?' + str(query)
            return TDSCatalog(url)
        except ET.ParseError:
            raise BadQueryError(self.get_catalog_raw(query))