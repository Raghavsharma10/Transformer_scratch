def post(self, url, data=None, **kwargs):
        """Encapsulte requests.post to use this class instance header"""
        return requests.post(url, data=data, headers=self.add_headers(**kwargs))