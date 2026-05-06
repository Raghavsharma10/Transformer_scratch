def emojis(self):
        """Retrieves a dictionary of all of the emojis that GitHub supports.

        :returns: dictionary where the key is what would be in between the
            colons and the value is the URL to the image, e.g., ::

                {
                    '+1': 'https://github.global.ssl.fastly.net/images/...',
                    # ...
                }
        """
        url = self._build_url('emojis')
        return self._json(self._get(url), 200)