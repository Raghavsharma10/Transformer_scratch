def wrap_all(self, rows: Iterable[Union[Mapping[str, Any], Sequence[Any]]]):
        """Return row tuple for each row in rows."""
        return (self.wrap(r) for r in rows)