def ping(self):
        """Return true if the server successfully pinged"""

        randomToken = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for x in range(32))

        r = self.doQuery('ping?data=' + randomToken)

        if r.status_code == 200:  # Query ok ?
            if r.json()['data'] == randomToken:  # Token equal ?
                return True
        return False