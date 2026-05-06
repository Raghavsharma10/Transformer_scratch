def _search_generator(self, item: Any, reverse: bool = False) -> Generator[Any, None, None]:
        """A helper method for `self.search` that returns a generator rather than a list."""
        results = 0
        for _, x in self.enumerate(item, reverse=reverse):
            yield x
            results += 1
        if results == 0:
            raise SearchError(str(item))