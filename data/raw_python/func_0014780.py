def _make_request_to_server(self, query_function, raise_for_status=True,
                                time_limit_seconds=2, retry_delay_seconds=0.2):
        """Retry sending request until timeout or until receiving a response.
        """
        start_time = datetime.datetime.now()
        while datetime.datetime.now() - start_time < datetime.timedelta(
                0, time_limit_seconds):
            error = None
            response = None
            try:
                response = query_function()
            except requests.exceptions.ConnectionError as e:
                error = ServerConnectionError(
                    "No response from server.\n%s" % e)
            except:
                if response:
                    logger.info(response.text)
                raise
            if response is not None and raise_for_status:
                # raises requests.exceptions.HTTPError
                self._raise_for_status(response)
            if error:
                time.sleep(retry_delay_seconds)
                continue
            else:
                return response
        raise error