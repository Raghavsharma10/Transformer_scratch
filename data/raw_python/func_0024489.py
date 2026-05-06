def crawl_cmd(self, seed_list, n):
        '''
        Runs the crawl job for n rounds
        :param seed_list: lines of seed URLs
        :param n: number of rounds
        :return: number of successful rounds
        '''

        print("Num Rounds "+str(n))

        cc = self.proxy.Crawl(seed=seed_list, rounds=n)
        rounds = cc.waitAll()
        print("Completed %d rounds" % len(rounds))
        return len(rounds)