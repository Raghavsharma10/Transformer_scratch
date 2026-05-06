def parse_resource_from_url(self, url):
        """
        Returns the appropriate resource name for the given URL.

        :param url:  API URL stub, like: '/api/hosts'
        :return: Resource name, like 'hosts', or None if not found
        """
        # special case for the api root
        if url == '/api':
            return 'api'
        elif url == '/katello':
            return 'katello'

        match = self.resource_pattern.match(url)
        if match:
            return match.groupdict().get('resource', None)