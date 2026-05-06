def remove_metric(self, metric_id):
        """
        Remove a metric from a monitored machine

        :param metric_id: Metric_id (provided by self.get_stats() )
        """
        payload = {
            'metric_id': metric_id
        }

        data = json.dumps(payload)

        req = self.request(self.mist_client.uri+"/clouds/"+self.cloud.id+"/machines/"+self.id+"/metrics", data=data)
        req.delete()