def reboot_droplet(self, droplet_id):
        """
        This method allows you to reboot a droplet. This is the preferred method
        to use if a server is not responding.
        """
        if not droplet_id:
            raise DOPException('droplet_id is required to reboot a droplet!')
        json = self.request('/droplets/%s/reboot' % droplet_id, method='GET')
        status = json.get('status')
        if status == 'OK':
            return json.get('event_id')
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))