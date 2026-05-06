def get_media_formats(self, media_id):
        """CR doesn't seem to provide the video_format and video_quality params
        through any of the APIs so we have to scrape the video page
        """
        url = (SCRAPER.API_URL + 'media-' + media_id).format(
            protocol=SCRAPER.PROTOCOL_INSECURE)
        format_pattern = re.compile(SCRAPER.VIDEO.FORMAT_PATTERN)
        formats = {}

        for format, param in iteritems(SCRAPER.VIDEO.FORMAT_PARAMS):
            resp = self._connector.get(url, params={param: '1'})
            if not resp.ok:
                continue
            try:
                match = format_pattern.search(resp.content)
            except TypeError:
                match = format_pattern.search(resp.text)
            if match:
                formats[format] = (int(match.group(1)), int(match.group(2)))
        return formats