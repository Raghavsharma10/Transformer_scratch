def get_record_rdd_from_json_files(self,
                                       json_files: List[str],
                                       data_processor: DataProcessor = SimpleJsonDataProcessor(),
                                       spark_session: Optional['SparkSession'] = None) -> 'RDD':
        """
        Reads the data from the given json_files path and converts them into the `Record`s format for
        processing. `data_processor` is used to process the per event data in those files to convert
        them into `Record`.

        :param json_files: List of json file paths. Regular Spark path wildcards are accepted.
        :param data_processor: `DataProcessor` to process each event in the json files.
        :param spark_session: `SparkSession` to use for execution. If None is provided then a basic
            `SparkSession` is created.
        :return: RDD containing Tuple[Identity, List[TimeAndRecord]] which can be used in
            `execute()`
        """
        spark_context = get_spark_session(spark_session).sparkContext
        raw_records: 'RDD' = spark_context.union(
            [spark_context.textFile(file) for file in json_files])
        return raw_records.mapPartitions(
            lambda x: self.get_per_identity_records(x, data_processor)).groupByKey().mapValues(list)