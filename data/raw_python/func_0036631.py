def _machine_actions(self, action):
        """
        Actions for the machine (e.g. stop, start etc)

        :param action: Can be "reboot", "start", "stop", "destroy"
        :returns: An updated list of the added machines
        """
        payload = {
            'action': action
        }
        data = json.dumps(payload)
        req = self.request(self.mist_client.uri+'/clouds/'+self.cloud.id+'/machines/'+self.id, data=data)
        req.post()
        self.cloud.update_machines()