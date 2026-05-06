def autoreconnect(self, sleep=1, attempt=3, exponential=True, jitter=5):
        """
        Tries to reconnect with some delay:

        exponential=False: up to `attempt` times with `sleep` seconds between
        each try

        exponential=True: up to `attempt` times with exponential growing `sleep`
        and random delay in range 1..`jitter` (exponential backoff)


        :param sleep: time to sleep between two attempts to reconnect
        :type sleep: float or int
        :param attempt: maximal number of attempts
        :type attempt: int
        :param exponential: if set - use exponential backoff logic
        :type exponential: bool
        :param jitter: top value of random delay, sec
        :type jitter: int

        """

        p = 0

        while attempt is None or attempt > 0:
            try:
                self.reconnect()
                return True
            except GraphiteSendException:

                if exponential:
                    p += 1
                    time.sleep(pow(sleep, p) + random.randint(1, jitter))
                else:
                    time.sleep(sleep)

                attempt -= 1

        return False