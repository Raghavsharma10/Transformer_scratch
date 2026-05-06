def get_record_rdd_from_rdd(
            self,
            rdd: 'RDD',
            data_processor: DataProcessor = SimpleDictionaryDataProcessor(),
    ) -> 'RDD':
        """
        Converts a RDD of raw events into the `Record`s format for processing. `data_processor` is
        used to process the per row data to convert them into `Record`.

        :param rdd: RDD containing the raw events.
        :param data_processor: `DataProcessor` to process each row in the given `rdd`.
        :return: RDD containing Tuple[Identity, List[TimeAndRecord]] which can be used in
            `execute()`
        """
        return rdd.mapPartitions(
            lambda x: self.get_per_identity_records(x, data_processor)).groupByKey().mapValues(list)