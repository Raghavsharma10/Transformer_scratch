def add(self, registry):
        """ Add works like replace, but only previously pushed metrics with the
            same name (and the same job and instance) will be replaced.
            (It uses HTTP method 'POST' to push to the Pushgateway.)
        """
        # POST
        payload = self.formatter.marshall(registry)
        r = requests.post(self.path, data=payload, headers=self.headers)