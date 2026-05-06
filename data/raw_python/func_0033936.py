def droplets(self):
        """
        This method returns the list of droplets
        """
        json = self.request('/droplets/', method='GET')
        status = json.get('status')
        if status == 'OK':
            droplet_json = json.get('droplets', [])
            droplets = [Droplet.from_json(droplet) for droplet in droplet_json]
            return droplets
        else:
            message = json.get('message', None)
            raise DOPException('[%s]: %s' % (status, message))