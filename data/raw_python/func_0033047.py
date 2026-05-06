def rebuild(self, image):
        """
        Rebuild the droplet with the specified image

            A rebuild action functions just like a new create. [APIDocs]_

        :param image: an image ID, an image slug, or an `Image` object
            representing the image the droplet should use as a base
        :type image: integer, string, or `Image`
        :return: an `Action` representing the in-progress operation on the
            droplet
        :rtype: Action
        :raises DOAPIError: if the API endpoint replies with an error
        """
        if isinstance(image, Image):
            image = image.id
        return self.act(type='rebuild', image=image)