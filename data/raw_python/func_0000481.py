def get_embed_url(self):
        """ Get correct embed url for Youtube or Vimeo. """
        embed_url = None
        youtube_embed_url = 'https://www.youtube.com/embed/{}'
        vimeo_embed_url = 'https://player.vimeo.com/video/{}'

        # Get video ID from url.
        if re.match(YOUTUBE_URL_RE, self.url):
            embed_url = youtube_embed_url.format(re.match(YOUTUBE_URL_RE, self.url).group(2))
        if re.match(VIMEO_URL_RE, self.url):
            embed_url = vimeo_embed_url.format(re.match(VIMEO_URL_RE, self.url).group(3))
        return embed_url