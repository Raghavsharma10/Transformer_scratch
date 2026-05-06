def _do_poll_problems(self):
        """Poll the server for the status of a set of problems.

        Note:
            This method is always run inside of a daemon thread.
        """
        try:
            # grouped futures (all scheduled within _POLL_GROUP_TIMEFRAME)
            frame_futures = {}

            def task_done():
                self._poll_queue.task_done()

            def add(future):
                # add future to query frame_futures
                # returns: worker lives on?

                # `None` task signifies thread termination
                if future is None:
                    task_done()
                    return False

                if future.id not in frame_futures and not future.done():
                    frame_futures[future.id] = future
                else:
                    task_done()

                return True

            while True:
                frame_futures.clear()

                # blocking add first scheduled
                frame_earliest, future = self._poll_queue.get()
                if not add(future):
                    return

                # try grouping if scheduled within grouping timeframe
                while len(frame_futures) < self._STATUS_QUERY_SIZE:
                    try:
                        task = self._poll_queue.get_nowait()
                    except queue.Empty:
                        break

                    at, future = task
                    if at - frame_earliest <= self._POLL_GROUP_TIMEFRAME:
                        if not add(future):
                            return
                    else:
                        task_done()
                        self._poll_queue.put(task)
                        break

                # build a query string with ids of all futures in this frame
                ids = [future.id for future in frame_futures.values()]
                _LOGGER.debug("Polling for status of futures: %s", ids)
                query_string = 'problems/?id=' + ','.join(ids)

                # if futures were cancelled while `add`ing, skip empty frame
                if not ids:
                    continue

                # wait until `frame_earliest` before polling
                delay = frame_earliest - time.time()
                if delay > 0:
                    _LOGGER.debug("Pausing polling %.2f sec for futures: %s", delay, ids)
                    time.sleep(delay)
                else:
                    _LOGGER.trace("Skipping non-positive delay of %.2f sec", delay)

                try:
                    _LOGGER.trace("Executing poll API request")

                    try:
                        response = self.session.get(posixpath.join(self.endpoint, query_string))
                    except requests.exceptions.Timeout:
                        raise RequestTimeout

                    if response.status_code == 401:
                        raise SolverAuthenticationError()
                    response.raise_for_status()

                    statuses = response.json()
                    for status in statuses:
                        self._handle_problem_status(status, frame_futures[status['id']])

                except BaseException as exception:
                    if not isinstance(exception, SolverAuthenticationError):
                        exception = IOError(exception)

                    for id_ in frame_futures.keys():
                        frame_futures[id_]._set_error(IOError(exception), sys.exc_info())

                for id_ in frame_futures.keys():
                    task_done()

                time.sleep(0)

        except Exception as err:
            _LOGGER.exception(err)