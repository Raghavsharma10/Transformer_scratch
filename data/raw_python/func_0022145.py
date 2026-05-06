def modify_rest_retry(self, total=8, connect=None, read=None, redirect=None, status=None,
                          method_whitelist=urllib3.util.retry.Retry.DEFAULT_METHOD_WHITELIST, status_forcelist=None,
                          backoff_factor=0.705883, raise_on_redirect=True, raise_on_status=True,
                          respect_retry_after_header=True, adapter_url="https://"):
        """
        Modify retry parameters for the SDK's rest call object.

        Parameters are directly from and passed directly to `urllib3.util.retry.Retry`, and get applied directly to
        the underlying `requests.Session` object.

        Default retry with total=8 and backoff_factor=0.705883:
         - Try 1, 0 delay (0 total seconds)
         - Try 2, 0 delay (0 total seconds)
         - Try 3, 0.705883 delay (0.705883 total seconds)
         - Try 4, 1.411766 delay (2.117649 total seconds)
         - Try 5, 2.823532 delay (4.941181 total seconds)
         - Try 6, 5.647064 delay (10.588245 total seconds)
         - Try 7, 11.294128 delay (21.882373 total seconds)
         - Try 8, 22.588256 delay (44.470629 total seconds)
         - Try 9, 45.176512 delay (89.647141 total seconds)
         - Try 10, 90.353024 delay (180.000165 total seconds)

        **Parameters:**

          - **total:** int, Total number of retries to allow. Takes precedence over other counts.
          - **connect:** int, How many connection-related errors to retry on.
          - **read:** int, How many times to retry on read errors.
          - **redirect:** int, How many redirects to perform. loops.
          - **status:** int, How many times to retry on bad status codes.
          - **method_whitelist:** iterable, Set of uppercased HTTP method verbs that we should retry on.
          - **status_forcelist:** iterable, A set of integer HTTP status codes that we should force a retry on.
          - **backoff_factor:** float, A backoff factor to apply between attempts after the second try.
          - **raise_on_redirect:** bool, True = raise a MaxRetryError, False = return latest 3xx response.
          - **raise_on_status:** bool, Similar logic to ``raise_on_redirect`` but for status responses.
          - **respect_retry_after_header:** bool, Whether to respect Retry-After header on status codes.
          - **adapter_url:** string, URL match for these retry values (default `https://`)

        **Returns:** No return, mutates the session directly
        """
        # Cloudgenix responses with 502/504 are usually recoverable. Use them if no list specified.
        if status_forcelist is None:
            status_forcelist = (413, 429, 502, 503, 504)

        retry = urllib3.util.retry.Retry(total=total,
                                         connect=connect,
                                         read=read,
                                         redirect=redirect,
                                         status=status,
                                         method_whitelist=method_whitelist,
                                         status_forcelist=status_forcelist,
                                         backoff_factor=backoff_factor,
                                         raise_on_redirect=raise_on_redirect,
                                         raise_on_status=raise_on_status,
                                         respect_retry_after_header=respect_retry_after_header)
        adapter = requests.adapters.HTTPAdapter(max_retries=retry)
        self._session.mount(adapter_url, adapter)
        return