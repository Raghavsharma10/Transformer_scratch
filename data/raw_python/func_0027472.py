def probes(self, **kwargs):
        """Returns a list of all Pingdom probe servers

        Parameters:

            * limit -- Limits the number of returned probes to the specified
                quantity
                    Type: Integer

            * offset -- Offset for listing (requires limit).
                    Type: Integer
                    Default: 0

            * onlyactive -- Return only active probes
                    Type: Boolean
                    Default: False

            * includedeleted -- Include old probes that are no longer in use
                    Type: Boolean
                    Default: False

        Returned structure:
        [
            {
                'id'        : <Integer> Unique probe id
                'country'   : <String> Country
                'city'      : <String> City
                'name'      : <String> Name
                'active'    : <Boolean> True if probe is active
                'hostname'  : <String> DNS name
                'ip'        : <String> IP address
                'countryiso': <String> Country ISO code
            },
            ...
        ]
        """

        # Warn user about unhandled parameters
        for key in kwargs:
            if key not in ['limit', 'offset', 'onlyactive', 'includedeleted']:
                sys.stderr.write("'%s'" % key + ' is not a valid argument ' +
                                 'of probes()\n')

        return self.request("GET", "probes", kwargs).json()['probes']