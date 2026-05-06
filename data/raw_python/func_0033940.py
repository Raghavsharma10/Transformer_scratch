def restore_droplet(self, droplet_id, image_id):
        """
        This method allows you to restore a droplet with a previous image or snapshot.
        This will be a mirror copy of the image or snapshot to your droplet.
        Be sure you have backed up any necessary information prior to restore.

        Required parameters:

            droplet_id:
                Numeric, this is the id of your droplet that you want to snapshot

            image_id:
                Numeric, this is the id of the image you would like to use to
                restore your droplet with
        """
        if not droplet_id:
            raise DOPException('droplet_id is required to restore a droplet!')
        if not image_id:
            raise DOPException('image_id is required to rebuild a droplet!')
        params = {'image_id': image_id}
        json = self.request('/droplets/%s/restore' % droplet_id, method='GET',
                            params=params)
        status = json.get('status')
        if status == 'OK':
            return json.get('event_id')
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))