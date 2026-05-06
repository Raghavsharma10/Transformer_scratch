def _extract_id(self) -> str:
        """
        Get video_id needed to obtain the real_url of the video.

        Raises:
            VideoIdNotMatchedError: If video_id is not matched with regular expression.

        """
        match = re.match(self._VALID_URL, self.url)

        if match:
            return match.group('video_id')
        else:
            raise VideoIdNotMatchedError