def create(self, sid, seedList):
        """
        Create a new named (sid) Seed from a list of seed URLs

        :param sid: the name to assign to the new seed list
        :param seedList: the list of seeds to use
        :return: the created Seed object
        """

        seedUrl = lambda uid, url: {"id": uid, "url": url}

        if not isinstance(seedList,tuple):
            seedList = (seedList,)

        seedListData = {
            "id": "12345",
            "name": sid,
            "seedUrls": [seedUrl(uid, url) for uid, url in enumerate(seedList)]
        }

        # As per resolution of https://issues.apache.org/jira/browse/NUTCH-2123
        seedPath = self.server.call('post', "/seed/create", seedListData, TextAcceptHeader)
        new_seed = Seed(sid, seedPath, self.server)
        return new_seed