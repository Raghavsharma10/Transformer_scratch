def getDetails(self):
        """Update check details, returns dictionary of details"""

        response = self.pingdom.request('GET', 'checks/%s' % self.id)
        self.__addDetails__(response.json()['check'])
        return response.json()['check']