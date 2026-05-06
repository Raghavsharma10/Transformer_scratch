def _make_url(self, url_part, blueprint_prefix):
        """Create URL from blueprint_prefix, api prefix and resource url.

        This method is used to defer the construction of the final url in
        the case that the Api is created with a Blueprint.

        :param url_part: The part of the url the endpoint is registered with
        :param blueprint_prefix: The part of the url contributed by the
            blueprint.  Generally speaking, BlueprintSetupState.url_prefix
        """
        parts = (blueprint_prefix, self.prefix, url_part)
        return ''.join(_ for _ in parts if _)