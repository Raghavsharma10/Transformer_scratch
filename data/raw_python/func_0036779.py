def get_url(cls, url, uid, **kwargs):
        """
        Construct the URL for talking to an individual resource.

        http://myapi.com/api/resource/1

        Args:
            url: The url for this resource
            uid: The unique identifier for an individual resource
            kwargs: Additional keyword argueents
        returns:
            final_url: The URL for this individual resource
        """
        if uid:
            url = '{}/{}'.format(url, uid)
        else:
            url = url
        return cls._parse_url_and_validate(url)