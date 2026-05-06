def calculateDelay(self, at, delay):
        """
        Creates the delay from now til the specified start time, uses "at" if available.
        :param at: the start time in %a %b %d %H:%M:%S %Y format.
        :param delay: the delay from now til start.
        :return: the delay.
        """
        if at is not None:
            return max((datetime.strptime(at, DATETIME_FORMAT) - datetime.utcnow()).total_seconds(), 0)
        elif delay is not None:
            return delay
        else:
            return 0