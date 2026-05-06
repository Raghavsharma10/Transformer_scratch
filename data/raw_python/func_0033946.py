def destroy_image(self, image_id_or_slug):
        """
        This method allows you to destroy an image. There is no way to restore
        a deleted image so be careful and ensure your data is properly backed up.

        Required parameters

            image_id:
                Numeric, this is the id of the image you would like to destroy
        """

        if not image_id_or_slug:
            msg = 'image_id_or_slug is required to destroy an image!'
            raise DOPException(msg)

        json = self.request('/images/%s/destroy' % image_id_or_slug, method='GET')
        status = json.get('status')
        return status