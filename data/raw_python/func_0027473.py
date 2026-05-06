def traceroute(self, host, probeid):
        """Perform a traceroute to a specified target from a specified Pingdom
            probe.

            Provide hostname to check and probeid to check from

        Returned structure:
        {
            'result'           : <String> Traceroute output
            'probeid'          : <Integer> Probe identifier
            'probedescription' : <String> Probe description
        }
        """

        response = self.request('GET', 'traceroute', {'host': host,
                                                      'probeid': probeid})
        return response.json()['traceroute']