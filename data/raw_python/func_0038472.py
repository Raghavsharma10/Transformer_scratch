def get_embed_url(self, targeting=None, recirc=None):
        """gets a canonical path to an embedded iframe of the video from the hub

        :return: the path to create an embedded iframe of the video
        :rtype: str
        """
        url = getattr(settings, "VIDEOHUB_EMBED_URL", self.DEFAULT_VIDEOHUB_EMBED_URL)
        url = url.format(self.id)
        if targeting is not None:
            for k, v in sorted(targeting.items()):
                url += '&{0}={1}'.format(k, v)
        if recirc is not None:
            url += '&recirc={0}'.format(recirc)
        return url