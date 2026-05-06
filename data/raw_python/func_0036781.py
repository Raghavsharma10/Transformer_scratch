def _parse_url_and_validate(cls, url):
        """
        Recieves a URL string and validates it using urlparse.

        Args:
            url: A URL string
        Returns:
            parsed_url: A validated URL
        Raises:
            BadURLException
        """
        parsed_url = urlparse(url)
        if parsed_url.scheme and parsed_url.netloc:
            final_url = parsed_url.geturl()
        else:
            raise BadURLException
        return final_url