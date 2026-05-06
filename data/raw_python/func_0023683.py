def _parse_uri(uri):
        """Parse and validate MediaFire URI."""

        tokens = urlparse(uri)

        if tokens.netloc != '':
            logger.error("Invalid URI: %s", uri)
            raise ValueError("MediaFire URI format error: "
                             "host should be empty - mf:///path")

        if tokens.scheme != '' and tokens.scheme != URI_SCHEME:
            raise ValueError("MediaFire URI format error: "
                             "must start with 'mf:' or '/'")

        return posixpath.normpath(tokens.path)