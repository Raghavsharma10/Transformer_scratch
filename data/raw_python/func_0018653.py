def _read_data_header_line(self, stream, expected_header):
        """Read a data header line, ensuring that it starts with the expected header

        Parameters
        ----------
        stream : :obj:`StreamIO`
            Stream object containing the text to read

        expected_header : str, list of strs
            Expected header of the data header line
        """
        pos = stream.tell()
        expected_header = (
            [expected_header] if isinstance(expected_header, str) else expected_header
        )
        for exp_hd in expected_header:
            tokens = stream.readline().split()
            try:
                assert tokens[0] == exp_hd
                return tokens[1:]
            except AssertionError:
                stream.seek(pos)
                continue

        assertion_msg = "Expected a header token of {}, got {}".format(
            expected_header, tokens[0]
        )
        raise AssertionError(assertion_msg)