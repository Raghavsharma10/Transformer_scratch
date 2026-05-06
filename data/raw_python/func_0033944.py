def images(self, filter='global'):
        """
        This method returns all the available images that can be accessed by
        your client ID. You will have access to all public images by default,
        and any snapshots or backups that you have created in your own account.

        Optional parameters

            filter:
                String, either "my_images" or "global"
        """
        if filter and filter not in ('my_images', 'global'):
            raise DOPException('"filter" must be either "my_images" or "global"')
        params = {}
        if filter:
            params['filter'] = filter
        json = self.request('/images', method='GET', params=params)
        status = json.get('status')
        if status == 'OK':
            images_json = json.get('images', [])
            images = [Image.from_json(image) for image in images_json]
            return images
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))