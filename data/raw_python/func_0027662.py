def publishPublicReport(self):
        """Activate public report for this check.

        Returns status message"""

        response = self.pingdom.request('PUT', 'reports.public/%s' % self.id)
        return response.json()['message']