def delete(self):
        """Delete this email report"""

        response = self.pingdom.request('DELETE',
                                        'reports.shared/%s' % self.id)
        return response.json()['message']