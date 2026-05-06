def to_table_data(self):
        """
        :raises ValueError:
        :raises pytablereader.error.ValidationError:
        """

        self._validate_source_data()

        for table_key, json_records in six.iteritems(self._buffer):
            headers = sorted(six.viewkeys(json_records))

            self._loader.inc_table_count()
            self._table_key = table_key

            yield TableData(
                self._make_table_name(),
                headers,
                zip(*[json_records.get(header) for header in headers]),
                dp_extractor=self._loader.dp_extractor,
                type_hints=self._extract_type_hints(headers),
            )