def create_tag(self, name, description=None, servers=[]):
        """
        Create a new Tag. Only name is mandatory.

        Returns the created Tag object.
        """
        servers = [str(server) for server in servers]
        body = {'tag': Tag(name, description, servers).to_dict()}
        res = self.request('POST', '/tag', body)

        return Tag(cloud_manager=self, **res['tag'])