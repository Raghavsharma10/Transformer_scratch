def destroy_droplet(self, droplet_id, scrub_data=False):
        """
        This method destroys one of your droplets - this is irreversible.

        Required parameters:

            droplet_id:
                Numeric, this is the id of your droplet that you want to destroy

        Optional parameters

            scrub_data:
                Boolean, this will strictly write 0s to your prior partition to
                ensure that all data is completely erased
        """
        params = {}

        if scrub_data:
            params['scrub_data'] = True

        json = self.request('/droplets/%s/destroy' % droplet_id, method='GET',
                            params=params)
        status = json.get('status')
        if status == 'OK':
            return json.get('event_id')
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))