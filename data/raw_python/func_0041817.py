def _infinite_iterator(self):
        """this iterator wraps the "_basic_iterator" when the configuration
        specifies that the "number_of_submissions" is set to "forever".
        Whenever the "_basic_iterator" is exhausted, it is called again to
        restart the iteration.  It is up to the implementation of the innermost
        iterator to define what starting over means.  Some iterators may
        repeat exactly what they did before, while others may iterate over
        new values"""
        while True:
            for crash_id in self._basic_iterator():
                if self._filter_disallowed_values(crash_id):
                    continue
                yield crash_id