def ping(self):
        """
        Posts the current state of each device to the server and schedules the next call in n seconds.
        :param serverURL:
        :return:
        """
        from datetime import datetime
        nextRun = datetime.utcnow().timestamp() + self.cfg.getPingInterval()
        self.sendHeartbeat()
        self.scheduleNextHeartbeat(nextRun)