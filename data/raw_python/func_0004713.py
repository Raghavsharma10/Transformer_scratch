def get(self, url, params=None, **kwargs):
        """Encapsulte requests.get to use this class instance header"""
        return requests.get(url, params=params, headers=self.add_headers(**kwargs))