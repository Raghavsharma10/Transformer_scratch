def get(self, *args, **kwargs):
        """
        Get a single object.

        This is a convenience wrapper for the search method that checks that
        only one object was returned, and returns that single object instead
        of a list.  This method takes the exact same arguments as search.
        """
        results = self.search(*args, **kwargs)
        num_results = len(results)
        if num_results == 1:
            return results[0]
        if num_results > 1:
            raise MultipleObjectsFound()
        raise ObjectNotFound()