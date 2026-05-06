def removePublicReport(self):
        """Deactivate public report for this check.

        Returns status message"""

        response = self.pingdom.request('DELETE',
                                        'reports.public/%s' % self.id)
        return response.json()['message']