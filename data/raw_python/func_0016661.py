def _poll(self, future):
        """Enqueue a problem to poll the server for status."""

        if future._poll_backoff is None:
            # on first poll, start with minimal back-off
            future._poll_backoff = self._POLL_BACKOFF_MIN

            # if we have ETA of results, schedule the first poll for then
            if future.eta_min and self._is_clock_diff_acceptable(future):
                at = datetime_to_timestamp(future.eta_min)
                _LOGGER.debug("Response ETA indicated and local clock reliable. "
                              "Scheduling first polling at +%.2f sec", at - epochnow())
            else:
                at = time.time() + future._poll_backoff
                _LOGGER.debug("Response ETA not indicated, or local clock unreliable. "
                              "Scheduling first polling at +%.2f sec", at - epochnow())

        else:
            # update exponential poll back-off, clipped to a range
            future._poll_backoff = \
                max(self._POLL_BACKOFF_MIN,
                    min(future._poll_backoff * 2, self._POLL_BACKOFF_MAX))

            # for poll priority we use timestamp of next scheduled poll
            at = time.time() + future._poll_backoff

        now = utcnow()
        future_age = (now - future.time_created).total_seconds()
        _LOGGER.debug("Polling scheduled at %.2f with %.2f sec new back-off for: %s (future's age: %.2f sec)",
                      at, future._poll_backoff, future.id, future_age)

        # don't enqueue for next poll if polling_timeout is exceeded by then
        future_age_on_next_poll = future_age + (at - datetime_to_timestamp(now))
        if self.polling_timeout is not None and future_age_on_next_poll > self.polling_timeout:
            _LOGGER.debug("Polling timeout exceeded before next poll: %.2f sec > %.2f sec, aborting polling!",
                          future_age_on_next_poll, self.polling_timeout)
            raise PollingTimeout

        self._poll_queue.put((at, future))