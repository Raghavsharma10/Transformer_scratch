def validate_query(self, query):
        """Validate a query.

        Determines whether `query` is well-formed. This includes checking for all
        required parameters, as well as checking parameters for valid values.

        Parameters
        ----------
        query : RadarQuery
            The query to validate

        Returns
        -------
        valid : bool
            Whether `query` is valid.

        """
        valid = True
        # Make sure all stations are in the table
        if 'stn' in query.spatial_query:
            valid = valid and all(stid in self.stations
                                  for stid in query.spatial_query['stn'])

        if query.var:
            valid = valid and all(var in self.variables for var in query.var)

        return valid