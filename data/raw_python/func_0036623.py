def enable(self):
        """
        Enable the Cloud.

        :returns:  A list of mist.clients' updated clouds.
        """
        payload = {
            "new_state": "1"
        }
        data = json.dumps(payload)
        req = self.request(self.mist_client.uri+'/clouds/'+self.id, data=data)
        req.post()
        self.enabled = True
        self.mist_client.update_clouds()