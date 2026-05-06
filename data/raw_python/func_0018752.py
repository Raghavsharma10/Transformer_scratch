def write_output_file(self,
                          path: str,
                          per_identity_data: 'RDD',
                          spark_session: Optional['SparkSession'] = None) -> None:
        """
        Basic helper function to persist data to disk.

        If window BTS was provided then the window BTS output to written in csv format, otherwise,
        the streaming BTS output is written in JSON format to the `path` provided

        :param path: Path where the output should be written.
        :param per_identity_data: Output of the `execute()` call.
        :param spark_session: `SparkSession` to use for execution. If None is provided then a basic
            `SparkSession` is created.
        :return:
        """
        _spark_session_ = get_spark_session(spark_session)
        if not self._window_bts:
            per_identity_data.flatMap(
                lambda x: [json.dumps(data, cls=BlurrJSONEncoder) for data in x[1][0].items()]
            ).saveAsTextFile(path)
        else:
            # Convert to a DataFrame first so that the data can be saved as a CSV
            _spark_session_.createDataFrame(per_identity_data.flatMap(lambda x: x[1][1])).write.csv(
                path, header=True)