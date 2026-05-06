def probes(self, fromtime, totime=None):
        """Get a list of probes that performed tests for a specified check
            during a specified period."""

        args = {'from': fromtime}
        if totime:
            args['to'] = totime

        response = self.pingdom.request('GET', 'summary.probes/%s' % self.id,
                                        args)

        return response.json()['probes']