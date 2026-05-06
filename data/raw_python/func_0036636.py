def add_metric(self, metric_id):
        """
        Add a metric to a monitored machine

        :param metric_id: Metric_id (provided by self.available_metrics)
        """
        payload = {
            'metric_id': metric_id
        }
        data = json.dumps(payload)
        req = self.request(self.mist_client.uri+"/clouds/"+self.cloud.id+"/machines/"+self.id+"/metrics", data=data)
        req.put()