def retry(self, retries, task_f, check_f=bool, wait_f=None):
        """
        Try a function up to n times.
        Raise an exception if it does not pass in time

        :param retries int: The number of times to retry
        :param task_f func: The function to be run and observed
        :param func()bool check_f: a function to check if task_f is complete
        :param func()bool wait_f: a function to run between checks
        """
        for attempt in range(retries):
            ret = task_f()
            if check_f(ret):
                return ret
            if attempt < retries - 1 and wait_f is not None:
                wait_f(attempt)
        raise RetryException("Giving up after {} failed attempt(s)".format(retries))