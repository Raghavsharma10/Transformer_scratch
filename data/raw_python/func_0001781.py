def _search_generator(self, item: Any) -> Generator[Tuple[Any, Any], None, None]:
        """A helper method for `self.search` that returns a generator rather than a list."""
        results = 0
        for key, value in self.enumerate(item):
            yield key, value
            results += 1
        if results == 0:
            raise SearchError(str(item))