def make_call(query, org_id, page):
    """
    Make a single UMAPI call with error handling and server-controlled throttling.
    (Adapted from sample code at https://www.adobe.io/products/usermanagement/docs/samples#retry)
    :param query: a query method from a UMAPI instance (callable as a function)
    :param org_id: the organization being queried
    :param page: the page number of the desired result set
    :return: the json (dictionary) received from the server (if any)
    """
    wait_time = 0
    num_attempts = 0

    while num_attempts < retry_max_attempts:
        if wait_time > 0:
            sleep(wait_time)
            wait_time = 0
        try:
            num_attempts += 1
            return query(org_id, page)
        except UMAPIRetryError as e:
            logger.warning("UMAPI service temporarily unavailable (attempt %d) -- %s", num_attempts, e.res.status_code)
            if "Retry-After" in e.res.headers:
                advice = e.res.headers["Retry-After"]
                advised_time = parsedate_tz(advice)
                if advised_time is not None:
                    # header contains date
                    wait_time = int(mktime_tz(advised_time) - time())
                else:
                    # header contains delta seconds
                    wait_time = int(advice)
            if wait_time <= 0:
                # use exponential back-off with random delay
                delay = randint(0, retry_random_delay_max)
                wait_time = (int(pow(2, num_attempts)) * retry_exponential_backoff_factor) + delay
            logger.warning("Next retry in %d seconds...", wait_time)
            continue
        except UMAPIRequestError as e:
            logger.warning("UMAPI error processing request -- %s", e.code)
            return {}
        except UMAPIError as e:
            logger.warning("HTTP error processing request -- %s: %s", e.res.status_code, e.res.text)
            return {}
    logger.error("UMAPI timeout...giving up on results page %d after %d attempts.", page, retry_max_attempts)
    return {}