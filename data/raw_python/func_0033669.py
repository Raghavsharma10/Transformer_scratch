def _retry(event, attempts, delay):
    """
    An iterator of pairs of (attempt number, event set), checking whether
    `event` is set up to `attempts` number of times, and delaying `delay`
    seconds in between.

    Terminates as soon as `event` is set, or until `attempts` have been made.

    Intended to be used in a loop, as in:

        for num, ok in _retry(event_to_wait_for, 10, 1.0):
            do_async_thing_that_sets_event()
            _log('tried %d time(s) to set event', num)
        if not ok:
            raise Exception('failed to set event')
    """
    event.clear()
    attempted = 0
    while attempted < attempts and not event.is_set():
        yield attempted, event.is_set()
        if event.wait(delay):
            break
    yield attempted, event.is_set()