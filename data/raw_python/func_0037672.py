def routeCoverage(self, msisdn):
        """
        If the route coverage lookup encounters an error, we will treat it as "not covered".
        """
        try:
            content = self.parseLegacy(self.request('utils/routeCoverage', {'msisdn': msisdn}))

            return {
                'routable': True,
                'destination': msisdn,
                'charge': float(content['Charge'])
            }
        except Exception:
            # If we encounter any error, we will treat it like it's "not covered"
            # TODO perhaps catch different types of exceptions so we can isolate certain global exceptions
            # like authentication
            return {
                'routable': False,
                'destination': msisdn,
                'charge': 0
            }