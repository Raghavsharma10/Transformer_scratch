def available_metrics(self):
        """
        List all available metrics that you can add to this machine

        :returns: A list of dicts, each of which is a metric that you can add to a monitored machine
        """
        req = self.request(self.mist_client.uri+"/clouds/"+self.cloud.id+"/machines/"+self.id+"/metrics")
        metrics = req.get().json()
        return metrics