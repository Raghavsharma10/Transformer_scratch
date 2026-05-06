def remove_tags(self, server, tags):
        """
        Remove tags from a server.

        - server: Server object or UUID string
        - tags: list of Tag objects or strings
        """
        uuid = str(server)
        tags = [str(tag) for tag in tags]

        url = '/server/{0}/untag/{1}'.format(uuid, ','.join(tags))
        return self.post_request(url)