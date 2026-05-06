def getChecks(self, **parameters):
        """Pulls all checks from pingdom

        Optional Parameters:

            * limit -- Limits the number of returned probes to the
                specified quantity.
                    Type: Integer (max 25000)
                    Default: 25000

            * offset -- Offset for listing (requires limit.)
                    Type: Integer
                    Default: 0

            * tags -- Filter listing by tag/s
                    Type: String
                    Default: None

        """

        # Warn user about unhandled parameters
        for key in parameters:
            if key not in ['limit', 'offset', 'tags']:
                sys.stderr.write('%s not a valid argument for getChecks()\n'
                                 % key)

        response = self.request('GET', 'checks', parameters)

        return [PingdomCheck(self, x) for x in response.json()['checks']]