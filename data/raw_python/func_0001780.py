def _search_generator(self, item: Any) -> Generator[Any, None, None]:
        """A helper method for `self.search` that returns a generator rather than a list."""
        results = 0
        for x in self.enumerate(item):
            yield x
            results += 1
        if results == 0:
            raise SearchError(str(item))