def replace(self, registry):
        """ Push triggers a metric collection and pushes all collected metrics
            to the Pushgateway specified by addr
            Note that all previously pushed metrics with the same job and
            instance will be replaced with the metrics pushed by this call.
            (It uses HTTP method 'PUT' to push to the Pushgateway.)
        """
        # PUT
        payload = self.formatter.marshall(registry)
        r = requests.put(self.path, data=payload, headers=self.headers)