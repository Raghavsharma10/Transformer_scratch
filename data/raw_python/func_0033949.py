def add_ssh_key(self, name, ssh_pub_key):
        """
        This method allows you to add a new public SSH key to your account.

        Required parameters

            name:
                String, the name you want to give this SSH key.

            ssh_pub_key:
                String, the actual public SSH key.
        """
        params = {'name': name, 'ssh_pub_key': ssh_pub_key}
        json = self.request('/ssh_keys/new', method='GET', params=params)
        status = json.get('status')
        if status == 'OK':
            ssh_key_json = json.get('ssh_key')
            ssh_key = SSHKey.from_json(ssh_key_json)
            return ssh_key
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))