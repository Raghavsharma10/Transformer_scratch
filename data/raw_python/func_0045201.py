def transform_source_rows(self):
        """
        Transforms the rows for the source system into (partial) dimensional data.
        """
        # Start timer for overall progress.
        self._time0 = time.perf_counter()

        self.pre_transform_source_rows()

        # Transform all source rows.
        with self._source_reader:
            with self._transformed_writer:
                with self._parked_writer:
                    with self._ignored_writer:
                        self._transform_rows()

        # Time end of transformation.
        self._time1 = time.perf_counter()

        # Load transformed rows into the fact table.
        self._load_transformed_rows()

        # Time end of loading transformed rows.
        self._time2 = time.perf_counter()

        # Load parked and ignored rows into the parked and ignored rows.
        self._load_ignored_rows()
        self._load_parked_rows()

        # Time end of loading parked and ignored rows.
        self._time3 = time.perf_counter()

        # Show statistics about number of rows and performance.
        self._log_statistics()