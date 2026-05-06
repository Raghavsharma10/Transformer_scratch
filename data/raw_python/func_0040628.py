def _get_iterator(self):
        """The iterator passed in can take several forms: a class that can be
        instantiated and then iterated over; a function that when called
        returns an iterator; an actual iterator/generator or an iterable
        collection.  This function sorts all that out and returns an iterator
        that can be used"""
        try:
            return self.job_param_source_iter(self.config)
        except TypeError:
            try:
                return self.job_param_source_iter()
            except TypeError:
                return self.job_param_source_iter