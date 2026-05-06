def send(self): # pragma: no cover
        """Send the draft."""
        response = self.session.request("method:queue", [ self.data ])
        self.data = response
        return self