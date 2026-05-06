def rename_droplet(self, droplet_id, name):
        """
        This method allows you to reinstall a droplet with a default image.
        This is useful if you want to start again but retain the same IP address
        for your droplet.

        Required parameters:

            droplet_id:
                Numeric, this is the id of your droplet that you want to snapshot

            image_id:
                Numeric, this is the id of the image you would like to use to
                rebuild  your droplet with
        """
        if not droplet_id:
            raise DOPException('droplet_id is required to rebuild a droplet!')
        if not name:
            raise DOPException('name is required to rebuild a droplet!')
        params = {'name': name}
        json = self.request('/droplets/%s/rename' % droplet_id, method='GET',
                            params=params)
        status = json.get('status')
        if status == 'OK':
            return json.get('event_id')
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))