def retry(method):
    """
    Allows to retry method execution few times.
    """

    def inner(self, *args, **kwargs):
        attempt_number = 1
        while attempt_number < self.retries:
            try:
                return method(self, *args, **kwargs)
            except HasOffersException as exc:
                if 'API usage exceeded rate limit' not in str(exc):
                    raise exc
                self.logger.debug('Retrying due: %s', exc)
                time.sleep(self.retry_timeout)
            except requests.exceptions.ConnectionError:
                # This happens when the session gets expired
                self.logger.debug('Recreating session due to ConnectionError')
                self._session = requests.Session()
            attempt_number += 1
        raise MaxRetriesExceeded

    return inner