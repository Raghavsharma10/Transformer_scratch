def get_hub_url(self):
        """gets a canonical path to the detail page of the video on the hub

        :return: the path to the consumer ui detail page of the video
        :rtype: str
        """
        url = getattr(settings, "VIDEOHUB_VIDEO_URL", self.DEFAULT_VIDEOHUB_VIDEO_URL)

        # slugify needs ascii
        ascii_title = ""
        if isinstance(self.title, str):
            ascii_title = self.title
        elif six.PY2 and isinstance(self.title, six.text_type):
            # Legacy unicode conversion
            ascii_title = self.title.encode('ascii', 'replace')

        path = slugify("{}-{}".format(ascii_title, self.id))

        return url.format(path)