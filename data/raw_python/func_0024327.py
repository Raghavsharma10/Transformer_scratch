def waitAll(self):
        """
        Execute all queued rounds and return when they have finished.

        If a job fails, a NutchCrawlException will be raised, with all completed jobs attached
        to the exception

        :return: a list of jobs completed for each round, organized by round (list-of-lists)
        """

        finishedRounds = [self.nextRound()]

        while self.currentRound < self.totalRounds:
            finishedRounds.append(self.nextRound())

        return finishedRounds