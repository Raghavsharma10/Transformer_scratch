def getEmailReports(self):
        """Returns a list of PingdomEmailReport instances."""

        reports = [PingdomEmailReport(self, x) for x in
                   self.request('GET',
                                'reports.email').json()['subscriptions']]

        return reports