def _parse_image(self):
        """
        Returns an instance of the image.Image class for the RSS feed.
        """
        image = {
            'title': self._channel.find('./image/title').text,
            'width': int(self._channel.find('./image/width').text),
            'height': int(self._channel.find('./image/height').text),
            'link': self._channel.find('./image/link').text,
            'url': self._channel.find('./image/url').text
        }

        return Image(image)