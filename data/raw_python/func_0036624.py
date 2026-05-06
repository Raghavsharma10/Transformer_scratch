def disable(self):
        """
        Disable the Cloud.

        :returns:  A list of mist.clients' updated clouds.
        """
        payload = {
            "new_state": "0"
        }
        data = json.dumps(payload)
        req = self.request(self.mist_client.uri+'/clouds/'+self.id, data=data)
        req.post()
        self.enabled = False
        self.mist_client.update_clouds()