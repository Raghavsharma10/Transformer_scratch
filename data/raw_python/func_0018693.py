def execute_per_identity_records(
            self,
            identity: str,
            records: List[TimeAndRecord],
            old_state: Optional[Dict[Key, Any]] = None) -> Tuple[str, Tuple[Dict, List]]:
        """
        Executes the streaming and window BTS on the given records. An option old state can provided
        which initializes the state for execution. This is useful for batch execution where the
        previous state is written out to storage and can be loaded for the next batch run.

        :param identity: Identity of the records.
        :param records: List of TimeAndRecord to be processed.
        :param old_state: Streaming BTS state dictionary from a previous execution.
        :return: Tuple[Identity, Tuple[Identity, Tuple[Streaming BTS state dictionary,
            List of window BTS output]].
        """
        schema_loader = SchemaLoader()
        if records:
            records.sort(key=lambda x: x[0])
        else:
            records = []

        block_data = self._execute_stream_bts(records, identity, schema_loader, old_state)
        window_data = self._execute_window_bts(identity, schema_loader)

        return identity, (block_data, window_data)