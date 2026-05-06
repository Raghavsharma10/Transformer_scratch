def averages(self, **kwargs):
        """Get the average time / uptime value for a specified check and time
            period.

        Optional parameters:

            * time_from -- Start time of period. Format is UNIX timestamp
                    Type: Integer
                    Default: 0

            * time_to -- End time of period. Format is UNIX timestamp
                    Type: Integer
                    Default: Current time

            * probes -- Filter to only use results from a list of probes.
                Format is a comma separated list of probe identifiers
                    Type: String
                    Default: All probes

            * includeuptime -- Include uptime information
                    Type: Boolean
                    Default: False

            * bycountry -- Split response times into country groups
                    Type: Boolean
                    Default: False

            * byprobe -- Split response times into probe groups
                    Type: Boolean
                    Default: False

        Returned structure:
        {
            'responsetime' :
            {
                'to'          : <Integer> Start time of period
                'from'        : <Integer> End time of period
                'avgresponse' : <Integer> Total average response time in
                                 milliseconds
            },
            < More can be included with optional parameters >
        }
        """

        # 'from' is a reserved word, use time_from instead
        if kwargs.get('time_from'):
            kwargs['from'] = kwargs.get('time_from')
            del kwargs['time_from']
        if kwargs.get('time_to'):
            kwargs['to'] = kwargs.get('time_to')
            del kwargs['time_to']

        # Warn user about unhandled parameters
        for key in kwargs:
            if key not in ['from', 'to', 'probes', 'includeuptime',
                           'bycountry', 'byprobe']:
                sys.stderr.write("'%s'" % key + ' is not a valid argument of' +
                                 '<PingdomCheck.averages()\n')

        response = self.pingdom.request('GET', 'summary.average/%s' % self.id,
                                        kwargs)

        return response.json()['summary']