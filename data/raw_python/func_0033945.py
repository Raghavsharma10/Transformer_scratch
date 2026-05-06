def show_image(self, image_id_or_slug):
        """
        This method displays the attributes of an image.

        Required parameters

            image_id:
                Numeric, this is the id of the image you would like to use to
                rebuild your droplet with
        """
        if not image_id_or_slug:
            msg = 'image_id_or_slug is required to destroy an image!'
            raise DOPException(msg)

        json = self.request('/images/%s' % image_id_or_slug, method='GET')
        image_json = json.get('image')
        status = json.get('status')
        if status == 'OK':
            image = Image.from_json(image_json)
            return image
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))