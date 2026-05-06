def schedule(self, duration, at=None, delay=None, callback=None):
        """
        schedules the measurement (to execute asynchronously).
        :param duration: how long to run for.
        :param at: the time to start at.
        :param delay: the time to wait til starting (use at or delay).
        :param callback: a callback.
        :return: nothing.
        """
        delay = self.calculateDelay(at, delay)
        self.callback = callback
        logger.info('Initiating measurement ' + self.name + ' for ' + str(duration) + 's in ' + str(delay) + 's')
        self.statuses.append({'name': ScheduledMeasurementStatus.SCHEDULED.name, 'time': datetime.utcnow()})
        threading.Timer(delay, self.execute, [duration]).start()