def destroy_ssh_key(self, ssh_key_id):
        """
        This method will delete the SSH key from your account.
        """
        json = self.request('/ssh_keys/%s/destroy' % ssh_key_id, method='GET')
        status = json.get('status')
        return status