def add_or_update_records(cls, tables: I2B2Tables, records: List["ObservationFact"]) -> Tuple[int, int]:
        """
        Add or update the observation_fact table as needed to reflect the contents of records
        :param tables: i2b2 sql connection
        :param records: records to apply
        :return: number of records added / modified
        """
        return cls._add_or_update_records(tables.crc_connection, tables.observation_fact, records)