def make_call(self, path, body=None, delete=False):
        """
        Make a single UMAPI call with error handling and retry on temporary failure.
        :param path: the string endpoint path for the call
        :param body: (optional) list of dictionaries to be serialized into the request body
        :return: the requests.result object (on 200 response), raise error otherwise
        """
        if body:
            request_body = json.dumps(body)
            def call():
                return self.session.post(self.endpoint + path, auth=self.auth, data=request_body, timeout=self.timeout)
        else:
            if not delete:
                def call():
                    return self.session.get(self.endpoint + path, auth=self.auth, timeout=self.timeout)
            else:
                def call():
                    return self.session.delete(self.endpoint + path, auth=self.auth, timeout=self.timeout)

        start_time = time()
        result = None
        for num_attempts in range(1, self.retry_max_attempts + 1):
            try:
                result = call()
                if result.status_code in [200,201,204]:
                    return result
                elif result.status_code in [429, 502, 503, 504]:
                    if self.logger: self.logger.warning("UMAPI timeout...service unavailable (code %d on try %d)",
                                                        result.status_code, num_attempts)
                    retry_wait = 0
                    if "Retry-After" in result.headers:
                        advice = result.headers["Retry-After"]
                        advised_time = parsedate_tz(advice)
                        if advised_time is not None:
                            # header contains date
                            retry_wait = int(mktime_tz(advised_time) - time())
                        else:
                            # header contains delta seconds
                            retry_wait = int(advice)
                    if retry_wait <= 0:
                        # use exponential back-off with random delay
                        delay = randint(0, self.retry_random_delay)
                        retry_wait = (int(pow(2, num_attempts - 1)) * self.retry_first_delay) + delay
                elif 201 <= result.status_code < 400:
                    raise ClientError("Unexpected HTTP Status {:d}: {}".format(result.status_code, result.text), result)
                elif 400 <= result.status_code < 500:
                    raise RequestError(result)
                else:
                    raise ServerError(result)
            except requests.Timeout:
                if self.logger: self.logger.warning("UMAPI connection timeout...(%d seconds on try %d)",
                                                    self.timeout, num_attempts)
                retry_wait = 0
                result = None
            if num_attempts < self.retry_max_attempts:
                if retry_wait > 0:
                    if self.logger: self.logger.warning("Next retry in %d seconds...", retry_wait)
                    sleep(retry_wait)
                else:
                    if self.logger: self.logger.warning("Immediate retry...")
        total_time = int(time() - start_time)
        if self.logger: self.logger.error("UMAPI timeout...giving up after %d attempts (%d seconds).",
                                          self.retry_max_attempts, total_time)
        raise UnavailableError(self.retry_max_attempts, total_time, result)