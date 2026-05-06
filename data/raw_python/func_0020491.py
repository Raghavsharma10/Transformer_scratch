def stream_logs(self, build_id):
        """
        stream logs from build

        :param build_id: str
        :return: iterator
        """
        kwargs = {'follow': 1}

        # If connection is closed within this many seconds, give up:
        min_idle_timeout = 60

        # Stream logs, but be careful of the connection closing
        # due to idle timeout. In that case, try again until the
        # call returns more quickly than a reasonable timeout
        # would be set to.
        last_activity = time.time()
        while True:
            buildlogs_url = self._build_url("builds/%s/log/" % build_id,
                                            **kwargs)
            try:
                response = self._get(buildlogs_url, stream=1,
                                     headers={'Connection': 'close'})
                check_response(response)

                for line in response.iter_lines():
                    last_activity = time.time()
                    yield line
            # NOTE1: If self._get causes ChunkedEncodingError, ConnectionError,
            # or IncompleteRead to be raised, they'll be wrapped in
            # OsbsNetworkException or OsbsException
            # NOTE2: If iter_lines causes ChunkedEncodingError
            # or IncompleteRead to be raised, it'll simply be silenced.
            # NOTE3: An exception may be raised from
            # check_response(). In this case, exception will be
            # wrapped in OsbsException or OsbsNetworkException,
            # inspect cause to detect ConnectionError.
            except OsbsException as exc:
                if not isinstance(exc.cause, ConnectionError):
                    raise

            idle = time.time() - last_activity
            logger.debug("connection closed after %ds", idle)
            if idle < min_idle_timeout:
                # Finish output
                return

            since = int(idle - 1)
            logger.debug("fetching logs starting from %ds ago", since)
            kwargs['sinceSeconds'] = since