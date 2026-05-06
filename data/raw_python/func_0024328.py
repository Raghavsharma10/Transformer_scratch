def Jobs(self, crawlId=None):
        """
        Create a JobClient for listing and creating jobs.
        The JobClient inherits the confId from the Nutch client.

        :param crawlId: crawlIds to use for this client.  If not provided, will be generated
         by nutch.defaultCrawlId()
        :return: a JobClient
        """
        crawlId = crawlId if crawlId else defaultCrawlId()
        return JobClient(self.server, crawlId, self.confId)