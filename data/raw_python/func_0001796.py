def get_status(self):
        """Get the AmCAT status page"""
        url = URL.status.format(**locals())
        return self.get_request(url)