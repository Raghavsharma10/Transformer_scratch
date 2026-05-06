def fetch(self):
        """Submit the request to the ACS Zeropoints Calculator.

        This method will:

        * submit the request
        * parse the response
        * format the results into a table with the correct units

        Returns
        -------
        tab : `astropy.table.QTable` or `None`
            If the request was successful, returns a table; otherwise, `None`.

        """
        LOG.info('Checking inputs...')
        valid_inputs = self._check_inputs()

        if valid_inputs:
            LOG.info('Submitting request to {}'.format(self._url))
            self._submit_request()
            if self._failed:
                return

            LOG.info('Parsing the response and formatting the results...')
            self._parse_and_format()
            return self.zpt_table

        LOG.error('Please fix the incorrect input(s)')