def get_per_identity_records(self, events: Iterable, data_processor: DataProcessor
                                 ) -> Generator[Tuple[str, TimeAndRecord], None, None]:
        """
        Uses the given iteratable events and the data processor convert the event into a list of
        Records along with its identity and time.
        :param events: iteratable events.
        :param data_processor: DataProcessor to process each event in events.
        :return: yields Tuple[Identity, TimeAndRecord] for all Records in events,
        """
        schema_loader = SchemaLoader()
        stream_bts_name = schema_loader.add_schema_spec(self._stream_bts)
        stream_transformer_schema: StreamingTransformerSchema = schema_loader.get_schema_object(
            stream_bts_name)
        for event in events:
            try:
                for record in data_processor.process_data(event):
                    try:
                        id = stream_transformer_schema.get_identity(record)
                        time = stream_transformer_schema.get_time(record)
                        yield (id, (time, record))
                    except Exception as err:
                        logging.error('{} in parsing Record {}.'.format(err, record))
            except Exception as err:
                logging.error('{} in parsing Event {}.'.format(err, event))