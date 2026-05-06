def resize_droplet(self, droplet_id, size):
        """
        This method allows you to resize a specific droplet to a different size.
        This will affect the number of processors and memory allocated to the droplet.

        Required parameters:

            droplet_id:
                Integer, this is the id of your droplet that you want to resize

            size, one of
                size_id: Numeric, this is the id of the size with which you
                         would like the droplet created
                size_slug: String, this is the slug of the size with which you
                           would like the droplet created
        """
        if not droplet_id:
            raise DOPException('droplet_id is required to resize a droplet!')
        params = {}
        size_id = size.get('size_id')
        if size_id:
            params.update({'size_id': size_id})
        else:
            size_slug = size.get('size_slug')
            if size_slug:
                params.update({'size_slug': size_slug})
            else:
                msg = 'size_id or size_slug are required to resize a droplet!'
                raise DOPException(msg)

        json = self.request('/droplets/%s/resize' % droplet_id, method='GET',
                            params=params)
        status = json.get('status')
        if status == 'OK':
            return json.get('event_id')
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))