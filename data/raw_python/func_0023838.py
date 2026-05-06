def get_servers(self, populate=False, tags_has_one=None, tags_has_all=None):
        """
        Return a list of (populated or unpopulated) Server instances.

        - populate = False (default) => 1 API request, returns unpopulated Server instances.
        - populate = True => Does 1 + n API requests (n = # of servers),
                             returns populated Server instances.

        New in 0.3.0: the list can be filtered with tags:
        - tags_has_one: list of Tag objects or strings
          returns servers that have at least one of the given tags

        - tags_has_all: list of Tag objects or strings
          returns servers that have all of the tags
        """
        if tags_has_all and tags_has_one:
            raise Exception('only one of (tags_has_all, tags_has_one) is allowed.')

        request = '/server'
        if tags_has_all:
            tags_has_all = [str(tag) for tag in tags_has_all]
            taglist = ':'.join(tags_has_all)
            request = '/server/tag/{0}'.format(taglist)

        if tags_has_one:
            tags_has_one = [str(tag) for tag in tags_has_one]
            taglist = ','.join(tags_has_one)
            request = '/server/tag/{0}'.format(taglist)

        servers = self.get_request(request)['servers']['server']

        server_list = list()
        for server in servers:
            server_list.append(Server(server, cloud_manager=self))

        if populate:
            for server_instance in server_list:
                server_instance.populate()

        return server_list