def get_full_url(self, parsed_url):
        """ Returns url path with querystring """
        full_path = parsed_url.path
        if parsed_url.query:
            full_path = '%s?%s' % (full_path, parsed_url.query)
        return full_path