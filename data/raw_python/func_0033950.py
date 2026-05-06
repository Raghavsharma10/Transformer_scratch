def show_ssh_key(self, ssh_key_id):
        """
        This method shows a specific public SSH key in your account that can be
        added to a droplet.
        """
        params = {}
        json = self.request('/ssh_keys/%s' % ssh_key_id, method='GET', params=params)
        status = json.get('status')
        if status == 'OK':
            ssh_key_json = json.get('ssh_key')
            ssh_key = SSHKey.from_json(ssh_key_json)
            return ssh_key
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))