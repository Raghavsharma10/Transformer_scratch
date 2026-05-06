def read_tuple(self):
        """Read a tuple from the pipe to Storm."""
        cmd = self.read_command()
        source = cmd["comp"]
        stream = cmd["stream"]
        values = cmd["tuple"]
        val_type = self._source_tuple_types[source].get(stream)
        return Tuple(
            cmd["id"],
            source,
            stream,
            cmd["task"],
            tuple(values) if val_type is None else val_type(*values),
        )