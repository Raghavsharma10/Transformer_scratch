def _toggle_monitoring(self, action, no_ssh=False):
        """
        Enable or disable monitoring on a machine

        :param action: Can be either "enable" or "disable"
        """
        payload = {
            'action': action,
            'name': self.name,
            'no_ssh': no_ssh,
            'public_ips': self.info['public_ips'],
            'dns_name': self.info['extra'].get('dns_name', 'n/a')
        }

        data = json.dumps(payload)

        req = self.request(self.mist_client.uri+"/clouds/"+self.cloud.id+"/machines/"+self.id+"/monitoring",
                           data=data)
        req.post()