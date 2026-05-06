def scheduleNextHeartbeat(self, nextRun):
        """
        Schedules the next ping.
        :param nextRun: when we should run next.
        :param serverURL: the URL to ping.
        :return:
        """
        import threading
        from datetime import datetime
        tilNextTime = max(nextRun - datetime.utcnow().timestamp(), 0)
        logging.getLogger('recorder').info("Scheduling next ping in " + str(round(tilNextTime, 3)) + " seconds")
        threading.Timer(tilNextTime, self.ping).start()