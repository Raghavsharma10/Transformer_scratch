def transfer_image(self, image_id_or_slug, region_id):
        """
        This method allows you to transfer an image to a specified region.

        Required parameters

            image_id:
                Numeric, this is the id of the image you would like to transfer.

            region_id
                Numeric, this is the id of the region to which you would like to transfer.
        """
        if not image_id_or_slug:
            msg = 'image_id_or_slug is required to transfer an image!'
            raise DOPException(msg)

        if not region_id:
            raise DOPException('region_id is required to transfer an image!')
        params = {'region_id': region_id}
        json = self.request('/images/%s/transfer' % image_id_or_slug,
                            method='GET', params=params)
        status = json.get('status')
        if status == 'OK':
            return json.get('event_id')
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))