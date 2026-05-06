def to_table_data(self):
        """
        :raises ValueError:
        :raises pytablereader.error.ValidationError:
        """

        self._validate_source_data()

        for table_key, json_records in six.iteritems(self._buffer):
            self._loader.inc_table_count()
            self._table_key = table_key

            yield TableData(
                self._make_table_name(),
                ["key", "value"],
                [record for record in json_records.items()],
                dp_extractor=self._loader.dp_extractor,
                type_hints=self._extract_type_hints(),
            )