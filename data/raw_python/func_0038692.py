def routeCoverage(self, msisdn):
        """
        If the route coverage lookup encounters an error, we will treat it as "not covered".
        """
        content = self.parseRest(self.request('rest/coverage/' + str(msisdn)))

        return {
            'routable': content['routable'],
            'destination': content['destination'].encode('utf-8'),
            'charge': float(content['minimumCharge'])
        }