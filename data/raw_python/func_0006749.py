def _call_retry(self, force_retry):
        """Call request and retry up to max_attempts times (or none if self.max_attempts=1)"""
        last_exception = None
        for i in range(self.max_attempts):
            try:
                log.info("Calling %s %s" % (self.method, self.url))
                response = self.requests_method(
                    self.url,
                    data=self.data,
                    params=self.params,
                    headers=self.headers,
                    timeout=(self.connect_timeout, self.read_timeout),
                    verify=self.verify_ssl,
                )

                if response is None:
                    log.warn("Got response None")
                    if self._method_is_safe_to_retry():
                        delay = 0.5 + i * 0.5
                        log.info("Waiting %s sec and Retrying since call is a %s" % (delay, self.method))
                        time.sleep(delay)
                        continue
                    else:
                        raise PyMacaronCoreException("Call %s %s returned empty response" % (self.method, self.url))

                return response

            except Exception as e:

                last_exception = e

                retry = force_retry

                if isinstance(e, ReadTimeout):
                    # Log enough to help debugging...
                    log.warn("Got a ReadTimeout calling %s %s" % (self.method, self.url))
                    log.warn("Exception was: %s" % str(e))
                    resp = e.response
                    if not resp:
                        log.info("Requests error has no response.")
                        # TODO: retry=True? Is it really safe?
                    else:
                        b = resp.content
                        log.info("Requests has a response with content: " + pprint.pformat(b))
                    if self._method_is_safe_to_retry():
                        # It is safe to retry
                        log.info("Retrying since call is a %s" % self.method)
                        retry = True

                elif isinstance(e, ConnectTimeout):
                    log.warn("Got a ConnectTimeout calling %s %s" % (self.method, self.url))
                    log.warn("Exception was: %s" % str(e))
                    # ConnectTimeouts are safe to retry whatever the call...
                    retry = True

                if retry:
                    continue
                else:
                    raise e

        # max_attempts has been reached: propagate the last received Exception
        if not last_exception:
            last_exception = Exception("Reached max-attempts (%s). Giving up calling %s %s" % (self.max_attempts, self.method, self.url))
        raise last_exception