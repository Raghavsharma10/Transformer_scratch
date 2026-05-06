def all_results(self):
        """
        Eagerly fetch all the results.
        This can be called after already doing some amount of iteration, and it will return
        all the previously-iterated results as well as any results that weren't yet iterated.
        :return: a list of all the results.
        """
        while not self._last_page_seen:
            self._next_page()
        self._next_item_index = len(self._results)
        return list(self._results)