def Crawl(self, seed, seedClient=None, jobClient=None, rounds=1, index=True):
        """
        Launch a crawl using the given seed
        :param seed: Type (Seed or SeedList) - used for crawl
        :param seedClient: if a SeedList is given, the SeedClient to upload, if None a default will be created
        :param jobClient: the JobClient to be used, if None a default will be created
        :param rounds: the number of rounds in the crawl
        :return: a CrawlClient to monitor and control the crawl
        """
        if seedClient is None:
            seedClient = self.Seeds()
        if jobClient is None:
            jobClient = self.Jobs()

        if type(seed) != Seed:
            seed = seedClient.create(jobClient.crawlId + '_seeds', seed)
        return CrawlClient(self.server, seed, jobClient, rounds, index)