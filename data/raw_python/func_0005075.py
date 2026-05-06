def get_url_base_from_url_token(
        self, url_api_token: str = "https://id.api.isogeo.com/oauth/token"
    ):
        """Returns the Isogeo API root URL (which is not included into
        credentials file) from the token URL (which is always included).

        :param url_api_token str: url to Isogeo API ID token generator
        """
        in_parsed = urlparse(url_api_token)
        api_url_base = in_parsed._replace(
            path="", netloc=in_parsed.netloc.replace("id.", "")
        )
        return api_url_base.geturl()