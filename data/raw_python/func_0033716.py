def checkVersion(self):
        """Check if the server use the same version of our protocol"""

        r = self.doQuery('version')

        if r.status_code == 200:  # Query ok ?
            data = r.json()

            if data['result'] == 'Ok' and data['version'] == self.PI_API_VERSION and data['protocol'] == self.PI_API_NAME:
                return True
        return False