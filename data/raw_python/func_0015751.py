def get_success_url(self):
        """
        Returns redirect URL for valid form submittal.

        :rtype: str.
        """
        if self.success_url:
            url = force_text(self.success_url)
        else:
            url = reverse('{0}:index'.format(self.url_namespace))

        return url