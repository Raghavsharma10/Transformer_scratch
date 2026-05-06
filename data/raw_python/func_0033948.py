def ssh_keys(self):
        """
        This method lists all the available public SSH keys in your account
        that can be added to a droplet.
        """
        params = {}
        json = self.request('/ssh_keys', method='GET', params=params)
        status = json.get('status')
        if status == 'OK':
            ssh_keys_json = json.get('ssh_keys', [])
            keys = [SSHKey.from_json(ssh_key) for ssh_key in ssh_keys_json]
            return keys
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))