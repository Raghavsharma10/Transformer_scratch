def _nextJob(self, job, nextRound=True):
        """
        Given a completed job, start the next job in the round, or return None

        :param nextRound: whether to start jobs from the next round if the current round is completed.
        :return: the newly started Job, or None if no job was started
        """

        jobInfo = job.info()
        assert jobInfo['state'] == 'FINISHED'

        roundEnd = False
        if jobInfo['type'] == 'INJECT':
            nextCommand = 'GENERATE'
        elif jobInfo['type'] == 'GENERATE':
            nextCommand = 'FETCH'
        elif jobInfo['type'] == 'FETCH':
            nextCommand = 'PARSE'
        elif jobInfo['type'] == 'PARSE':
            nextCommand = 'UPDATEDB'
        elif jobInfo['type'] == 'UPDATEDB':
            nextCommand = 'INVERTLINKS'
        elif jobInfo['type'] == 'INVERTLINKS':
            nextCommand = 'DEDUP'
        elif jobInfo['type'] == 'DEDUP':
            if self.enable_index:
                nextCommand = 'INDEX'
            else:
                roundEnd = True
        elif jobInfo['type'] == 'INDEX':
            roundEnd = True
        else:
            raise NutchException("Unrecognized job type {}".format(jobInfo['type']))

        if roundEnd:
            if nextRound and self.currentRound < self.totalRounds:
                nextCommand = 'GENERATE'
                self.currentRound += 1
            else:
                return None

        return self.jobClient.create(nextCommand)