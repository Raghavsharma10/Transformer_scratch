def restore(self, image):
        """
        Restore the droplet to the specified backup image

            A Droplet restoration will rebuild an image using a backup image.
            The image ID that is passed in must be a backup of the current
            Droplet instance.  The operation will leave any embedded SSH keys
            intact. [APIDocs]_

        :param image: an image ID, an image slug, or an `Image` object
            representing a backup image of the droplet
        :type image: integer, string, or `Image`
        :return: an `Action` representing the in-progress operation on the
            droplet
        :rtype: Action
        :raises DOAPIError: if the API endpoint replies with an error
        """
        if isinstance(image, Image):
            image = image.id
        return self.act(type='restore', image=image)