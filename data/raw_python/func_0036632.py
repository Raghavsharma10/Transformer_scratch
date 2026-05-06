def probe(self, key_id=None, ssh_user=None):
        """
        If no parameter is provided, mist.io will try to probe the machine with
        the default
        :param key_id: Optional. Give if you explicitly want to probe with this
        key_id
        :param ssh_user: Optional. Give if you explicitly want a specific user
        :returns: A list of data received by the probing (e.g. uptime etc)
        """
        ips = [ip for ip in self.info['public_ips'] if ':' not in ip]

        if not ips:
            raise Exception("No public IPv4 address available to connect to")
        payload = {
            'host': ips[0],
            'key': key_id,
            'ssh_user': ssh_user
        }
        data = json.dumps(payload)
        req = self.request(self.mist_client.uri + "/clouds/" + self.cloud.id +
                           "/machines/" + self.id + "/probe", data=data)
        probe_info = req.post().json()
        self.probed = True
        return probe_info