def rest_call(self, url, method, data=None, sensitive=False, timeout=None, content_json=True,
                  retry=None, max_retry=None, retry_sleep=None):
        """
        Generic REST call worker function

        **Parameters:**

          - **url:** URL for the REST call
          - **method:** METHOD for the REST call
          - **data:** Optional DATA for the call (for POST/PUT/etc.)
          - **sensitive:** Flag if content request/response should be hidden from logging functions
          - **timeout:** Requests Timeout
          - **content_json:** Bool on whether the Content-Type header should be set to application/json
          - **retry:** DEPRECATED - please use `cloudgenix.API.modify_rest_retry` instead.
          - **max_retry:** DEPRECATED - please use `cloudgenix.API.modify_rest_retry` instead.
          - **retry_sleep:** DEPRECATED - please use `cloudgenix.API.modify_rest_retry` instead.

        **Returns:** Requests.Response object, extended with:

          - **cgx_status**: Bool, True if a successful CloudGenix response, False if error.
          - **cgx_content**: Content of the response, guaranteed to be in Dict format. Empty/invalid responses
          will be converted to a Dict response.
        """
        # pull retry related items from Constructor if not specified.
        if timeout is None:
            timeout = self.rest_call_timeout
        if retry is not None:
            # Someone using deprecated retry code. Notify.
            sys.stderr.write("WARNING: 'retry' option of rest_call() has been deprecated. "
                             "Please use 'API.modify_rest_retry()' instead.")
        if max_retry is not None:
            # Someone using deprecated retry code. Notify.
            sys.stderr.write("WARNING: 'max_retry' option of rest_call() has been deprecated. "
                             "Please use 'API.modify_rest_retry()' instead.")
        if retry_sleep is not None:
            # Someone using deprecated retry code. Notify.
            sys.stderr.write("WARNING: 'max_retry' option of rest_call() has been deprecated. "
                             "Please use 'API.modify_rest_retry()' instead.")

        # Get logging level, use this to bypass logging functions with possible large content if not set.
        logger_level = api_logger.getEffectiveLevel()

        # populate headers and cookies from session.
        if content_json and method.lower() not in ['get', 'delete']:
            headers = {
                'Content-Type': 'application/json'
            }
        else:
            headers = {}

        # add session headers
        headers.update(self._session.headers)
        cookie = self._session.cookies.get_dict()

        # make sure data is populated if present.
        if isinstance(data, (list, dict)):
            data = json.dumps(data)

        api_logger.debug('REST_CALL URL = %s', url)

        # make request
        try:
            if not sensitive:
                api_logger.debug('\n\tREQUEST: %s %s\n\tHEADERS: %s\n\tCOOKIES: %s\n\tDATA: %s\n',
                                 method.upper(), url, headers, cookie, data)

            # Actual request
            response = self._session.request(method, url, data=data, verify=self.ca_verify_filename,
                                             stream=True, timeout=timeout, headers=headers, allow_redirects=False)

            # Request complete - lets parse.
            # if it's a non-CGX-good response, return with cgx_status = False
            if response.status_code not in [requests.codes.ok,
                                            requests.codes.no_content,
                                            requests.codes.found,
                                            requests.codes.moved]:

                # Simple JSON debug
                if not sensitive:
                    try:
                        api_logger.debug('RESPONSE HEADERS: %s\n', json.dumps(
                            json.loads(text_type(response.headers)), indent=4))
                    except ValueError:
                        api_logger.debug('RESPONSE HEADERS: %s\n', text_type(response.headers))
                    try:
                        api_logger.debug('RESPONSE: %s\n', json.dumps(response.json(), indent=4))
                    except ValueError:
                        api_logger.debug('RESPONSE: %s\n', text_type(response.text))
                else:
                    api_logger.debug('RESPONSE NOT LOGGED (sensitive content)')

                api_logger.debug("Error, non-200 response received: %s", response.status_code)

                # CGX extend requests.Response for return
                response.cgx_status = False
                response.cgx_content = self._catch_nonjson_streamresponse(response.text)
                return response

            else:

                # Simple JSON debug
                if not sensitive and (logger_level <= logging.DEBUG and logger_level != logging.NOTSET):
                    try:
                        api_logger.debug('RESPONSE HEADERS: %s\n', json.dumps(
                            json.loads(text_type(response.headers)), indent=4))
                        api_logger.debug('RESPONSE: %s\n', json.dumps(response.json(), indent=4))
                    except ValueError:
                        api_logger.debug('RESPONSE HEADERS: %s\n', text_type(response.headers))
                        api_logger.debug('RESPONSE: %s\n', text_type(response.text))
                elif sensitive:
                    api_logger.debug('RESPONSE NOT LOGGED (sensitive content)')

                # CGX extend requests.Response for return
                response.cgx_status = True
                response.cgx_content = self._catch_nonjson_streamresponse(response.text)
                return response

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, urllib3.exceptions.MaxRetryError)\
                as e:

            api_logger.info("Error, %s.", text_type(e))

            # make a requests.Response object for return since we didn't get one.
            response = requests.Response

            # CGX extend requests.Response for return
            response.cgx_status = False
            response.cgx_content = {
                '_error': [
                    {
                        'message': 'REST Request Exception: {}'.format(e),
                        'data': {},
                    }
                ]
            }
            return response