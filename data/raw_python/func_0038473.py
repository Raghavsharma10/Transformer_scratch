def get_api_url(self):
        """gets a canonical path to the api detail url of the video on the hub

        :return: the path to the api detail of the video
        :rtype: str
        """
        url = getattr(settings, 'VIDEOHUB_API_URL', None)
        # Support alternate setting (used by most client projects)
        if not url:
            url = getattr(settings, 'VIDEOHUB_API_BASE_URL', None)
            if url:
                url = url.rstrip('/') + '/videos/{}'
        if not url:
            url = self.DEFAULT_VIDEOHUB_API_URL
        return url.format(self.id)