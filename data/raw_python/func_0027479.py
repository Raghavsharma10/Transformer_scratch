def getSharedReports(self):
        """Returns a list of PingdomSharedReport instances"""

        response = self.request('GET',
                                'reports.shared').json()['shared']['banners']

        reports = [PingdomSharedReport(self, x) for x in response]
        return reports