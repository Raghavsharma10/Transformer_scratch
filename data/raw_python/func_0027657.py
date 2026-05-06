def getAnalyses(self, **kwargs):
        """Returns a list of the latest root cause analysis results for a
            specified check.

        Optional Parameters:

            * limit -- Limits the number of returned results to the
                specified quantity.
                    Type: Integer
                    Default: 100

            * offset -- Offset for listing. (Requires limit.)
                    Type: Integer
                    Default: 0

            * time_from -- Return only results with timestamp of first test greater
                or equal to this value. Format is UNIX timestamp.
                    Type: Integer
                    Default: 0

            * time_to -- Return only results with timestamp of first test less or
                equal to this value. Format is UNIX timestamp.
                    Type: Integer
                    Default: Current Time

        Returned structure:
        [
            {
                'id' : <Integer> Analysis id
                'timefirsttest'   : <Integer> Time of test that initiated the
                                             confirmation test
                'timeconfrimtest' : <Integer> Time of the confirmation test
                                               that perfromed the error
                                               analysis
            },
            ...
        ]
        """

        # 'from' is a reserved word, use time_from instead
        if kwargs.get('time_from'):
            kwargs['from'] = kwargs.get('time_from')
            del kwargs['time_from']
        if kwargs.get('time_to'):
            kwargs['to'] = kwargs.get('time_to')
            del kwargs['time_to']

        # Warn user about unhandled kwargs
        for key in kwargs:
            if key not in ['limit', 'offset', 'from', 'to']:
                sys.stderr.write('%s not a valid argument for analysis()\n'
                                 % key)

        response = self.pingdom.request('GET', 'analysis/%s' % self.id,
                                        kwargs)

        return [PingdomAnalysis(self, x) for x in response.json()['analysis']]