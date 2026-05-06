def _get_data(self):
        """Process the IGRA2 text file for observations at site_id matching time.

        Return:
        -------
            :class: `pandas.DataFrame` containing the body data.
            :class: `pandas.DataFrame` containing the header data.
        """
        # Split the list of times into begin and end dates. If only
        # one date is supplied, set both begin and end dates equal to that date.

        body, header, dates_long, dates = self._get_data_raw()

        params = self._get_fwf_params()

        df_body = pd.read_fwf(StringIO(body), **params['body'])
        df_header = pd.read_fwf(StringIO(header), **params['header'])
        df_body['date'] = dates_long

        df_body = self._clean_body_df(df_body)
        df_header = self._clean_header_df(df_header)
        df_header['date'] = dates

        return df_body, df_header