def patch(self, url, data=None, **kwargs):
        """Encapsulte requests.patch to use this class instance header"""
        return requests.patch(url, data=data, headers=self.add_headers(**kwargs))