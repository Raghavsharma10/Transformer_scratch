def params(self):
        """
        URL parameters for wq.io.loaders.NetLoader
        """
        params, complex = self.get_params()
        url_params = self.default_params.copy()
        url_params.update(self.serialize_params(params, complex))
        return url_params