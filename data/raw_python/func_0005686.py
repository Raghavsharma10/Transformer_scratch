def keys(self) -> Iterator[str]:
        """return all possible paths one can take from this ApiNode"""
        if self.param:
            yield self.param_name
        yield from self.paths.keys()