def _limited_iterator(self):
        """this is the iterator for the case when "number_of_submissions" is
        set to an integer.  It goes through the innermost iterator exactly the
        number of times specified by "number_of_submissions"  To do that, it
        might run the innermost iterator to exhaustion.  If that happens, that
        innermost iterator is called again to start over.  It is up to the
        implementation of the innermost iteration to define what starting
        over means.  Some iterators may repeat exactly what they did before,
        while others may iterate over new values"""
        i = 0
        while True:
            for crash_id in self._basic_iterator():
                if self._filter_disallowed_values(crash_id):
                    continue
                if crash_id is None:
                    # it's ok to yield None, however, we don't want it to
                    # be counted as a yielded value
                    yield crash_id
                    continue
                if i == int(self.config.number_of_submissions):
                    # break out of inner loop, abandoning the wrapped iter
                    break
                i += 1
                yield crash_id
            # repeat the quit test, to break out of the outer loop and
            # if necessary, prevent recycling the wrapped iter
            if i == int(self.config.number_of_submissions):
                break