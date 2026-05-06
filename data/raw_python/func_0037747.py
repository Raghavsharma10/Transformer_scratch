def Retry(target=None, args=[], kwargs={},
          options={"retry": True, "interval": 1}):
    """
    options
        retry
            True, infinity retries
            False, no retries
            Number, retries times
        interval
            time period for retry
        return
            None if no success
            Message if success
    """
    retry = options["retry"]
    interval = options["interval"]

    while True:
        try:
            resp = target(*args, **kwargs)
            # status error
            if resp.code == 200:
                return resp

            _logger.debug("Request got response status: %s"
                          % (resp.code,) + " retry: %s" % (retry,))
        except TimeoutError:
            _logger.debug("Request message is timeout")
            _logger.debug(args)
            _logger.debug(kwargs)

        # register unsuccessful goes here
        # infinity retry
        if retry is True:
            sleep(interval)
            continue

        # no retry
        if retry is False:
            return None

        # retrying
        try:
            retry = retry - 1
            if retry <= 0:
                return None
        except TypeError as e:
            raise e
        sleep(interval)