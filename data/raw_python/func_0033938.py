def power_cycle_droplet(self, droplet_id):
        """
        This method allows you to power cycle a droplet. This will turn off the
        droplet and then turn it back on.
        """
        if not droplet_id:
            msg = 'droplet_id is required to power cycle a droplet!'
            raise DOPException(msg)
        json = self.request('/droplets/%s/power_cycle' % droplet_id, method='GET')
        status = json.get('status')
        if status == 'OK':
            return json.get('event_id')
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))