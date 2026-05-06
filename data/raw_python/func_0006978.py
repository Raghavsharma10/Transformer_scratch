def wrap(self, row: Union[Mapping[str, Any], Sequence[Any]]):
        """Return row tuple for row."""
        return (
            self.dataclass(
                **{
                    ident: row[column_name]
                    for ident, column_name in self.ids_and_column_names.items()
                }
            )
            if isinstance(row, Mapping)
            else self.dataclass(
                **{ident: val for ident, val in zip(self.ids_and_column_names.keys(), row)}
            )
        )