def get_data(self, query):
        """Fetch parsed data from a THREDDS server using NCSS.

        Requests data from the NCSS endpoint given the parameters in `query` and
        handles parsing of the returned content based on the mimetype.

        Parameters
        ----------
        query : NCSSQuery
            The parameters to send to the NCSS endpoint

        Returns
        -------
        Parsed data response from the server. Exact format depends on the format of the
        response.

        See Also
        --------
        get_data_raw

        """
        resp = self.get_query(query)
        return response_handlers(resp, self.unit_handler)