def print_output(self, per_identity_data: 'RDD') -> None:
        """
        Basic helper function to write data to stdout. If window BTS was provided then the window
        BTS output is written, otherwise, the streaming BTS output is written to stdout.

        WARNING - For large datasets this will be extremely slow.

        :param per_identity_data: Output of the `execute()` call.
        """
        if not self._window_bts:
            data = per_identity_data.flatMap(
                lambda x: [json.dumps(data, cls=BlurrJSONEncoder) for data in x[1][0].items()])
        else:
            # Convert to a DataFrame first so that the data can be saved as a CSV
            data = per_identity_data.map(
                lambda x: json.dumps((x[0], x[1][1]), cls=BlurrJSONEncoder))
        for row in data.collect():
            print(row)